#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pandas as pd
import pytest

pytorch = pytest.importorskip("torch")
t4r = pytest.importorskip("transformers4rec")
tr = pytest.importorskip("transformers4rec.torch")

triton = pytest.importorskip("merlin.systems.triton")
data_conversions = pytest.importorskip("merlin.systems.triton.conversions")

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

from tritonclient.utils import np_to_triton_dtype  # noqa

from merlin.core.dispatch import make_df  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag import Ensemble  # noqa
from merlin.systems.dag.ops.pytorch import PredictPyTorch  # noqa
from merlin.systems.triton.utils import run_triton_server  # noqa

# from tests.unit.systems.utils.triton import _run_ensemble_on_tritonserver


def test_serve_t4r_with_torchscript(tmpdir):
    torch_yoochoose_like = tr.data.tabular_sequence_testing_data.torch_synthetic_data(
        num_rows=100, min_session_length=5, max_session_length=20
    )

    t4r_yoochoose_schema = t4r.data.tabular_sequence_testing_data.schema

    # TODO: This schema is some weird list thing, but it should be a Merlin schema.
    #       For now, let's convert it across in this test, but ultimately we should
    #       rework T4R to use Merlin Schemas.

    merlin_yoochoose_schema = Schema()
    for column in t4r_yoochoose_schema:
        name = column.name

        # The feature types in the T4R schemas are a bit hard to work with:
        # https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/538fc54bb8f2e3dc79224e497bebee15b00e4ab7/merlin_standard_lib/proto/schema_bp.py#L43-L53

        # Getting tritonclient.utils.InferenceServerException: [StatusCode.INVALID_ARGUMENT]
        # inference input data-type is 'FP32', model expects 'INT32' for 'ensemble_model'
        # which may (or may not) be related to the following line
        dtype = {0: np.int32, 2: np.int32, 3: np.float32}[column.type]
        tags = column.tags
        value_count = {"min": column.value_count.min, "max": column.value_count.max}
        is_list = bool(value_count)
        is_ragged = value_count.get("min", 0) != value_count.get("max", 0)
        int_domain = {"min": column.int_domain.min, "max": column.int_domain.max}
        properties = {"value_count": value_count, "int_domain": int_domain}
        col_schema = ColumnSchema(
            name,
            dtype=dtype,
            tags=tags,
            properties=properties,
            is_list=is_list,
            is_ragged=is_ragged,
        )
        merlin_yoochoose_schema[name] = col_schema

    input_module = t4r.torch.TabularSequenceFeatures.from_schema(
        t4r_yoochoose_schema,
        max_sequence_length=20,
        d_output=64,
        masking="causal",
    )
    prediction_task = t4r.torch.NextItemPredictionTask(hf_format=True, weight_tying=True)
    transformer_config = t4r.config.transformer.XLNetConfig.build(
        d_model=64, n_head=8, n_layer=2, total_seq_length=20
    )
    model = transformer_config.to_torch_model(input_module, prediction_task)

    _ = model(torch_yoochoose_like, training=False)

    model.eval()

    traced_model = pytorch.jit.trace(model, torch_yoochoose_like, strict=False)
    assert isinstance(traced_model, pytorch.jit.TopLevelTracedModule)
    assert pytorch.allclose(
        model(torch_yoochoose_like)["predictions"],
        traced_model(torch_yoochoose_like)["predictions"],
    )

    output_schema = Schema([ColumnSchema("output")])

    torch_op = merlin_yoochoose_schema.column_names >> PredictPyTorch(
        traced_model, merlin_yoochoose_schema, output_schema
    )

    ensemble = Ensemble(torch_op, merlin_yoochoose_schema)
    ens_config, node_configs = ensemble.export(str(tmpdir))

    df_cols = {}
    for name, tensor in torch_yoochoose_like.items():
        df_cols[name] = tensor.numpy().astype(merlin_yoochoose_schema[name].dtype)
        if len(tensor.shape) > 1:
            df_cols[name] = list(df_cols[name])

    df = make_df(df_cols)[merlin_yoochoose_schema.column_names].iloc[:3]

    # response = _run_ensemble_on_tritonserver(str(tmpdir), ["output"], df, ensemble.name)

    inputs = convert_df_to_triton_input(merlin_yoochoose_schema, df)

    outputs = [grpcclient.InferRequestedOutput(col) for col in output_schema.column_names]

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer("ensemble_model", inputs, outputs=outputs)

    assert response


def convert_df_to_triton_input(schema, batch, input_class=grpcclient.InferInput, dtype="int32"):
    columns = [(col_name, batch[col_name]) for col_name in schema.column_names]
    inputs = []
    for i, (col_name, col) in enumerate(columns):
        if schema[col_name].is_list and schema[col_name].is_ragged:
            if isinstance(col, pd.Series):
                raise ValueError("this function doesn't support CPU list values yet")
            inputs.append(
                _convert_column_to_triton_input(
                    col._column.offsets.values_host.astype(schema[col_name].dtype),
                    col_name + "__nnzs",
                    input_class,
                )
            )
            inputs.append(
                _convert_column_to_triton_input(
                    col.list.leaves.values_host.astype("int32"), col_name + "__values", input_class
                )
            )
        else:
            values = col.values if isinstance(col, pd.Series) else col.values_host
            inputs.append(_convert_column_to_triton_input(values, col_name, input_class))
    return inputs


def _convert_column_to_triton_input(col, name, input_class=grpcclient.InferInput):
    col = col.reshape(len(col), 1)
    dtype = np_to_triton_dtype(col.dtype)
    input_tensor = input_class(name, col.shape, dtype)
    input_tensor.set_data_from_numpy(col)
    return input_tensor
