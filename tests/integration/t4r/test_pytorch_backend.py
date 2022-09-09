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
import pytest

torch = pytest.importorskip("torch")
t4r = pytest.importorskip("transformers4rec")
tr = pytest.importorskip("transformers4rec.torch")

triton = pytest.importorskip("merlin.systems.triton")
data_conversions = pytest.importorskip("merlin.systems.triton.conversions")

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

from merlin.core.dispatch import make_df  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag import Ensemble  # noqa
from merlin.systems.dag.ops.pytorch import PredictPyTorch  # noqa
from merlin.systems.triton import convert_df_to_triton_input  # noqa
from merlin.systems.triton.utils import run_triton_server  # noqa

# TODO: Use this again once `convert_df_to_triton_input` has been reworked
# from tests.unit.systems.utils.triton import _run_ensemble_on_tritonserver


class ServingAdapter(torch.nn.Module):
    def __init__(self, model):
        super(ServingAdapter, self).__init__()

        self.model = model

    def forward(self, batch):
        return self.model(batch)["predictions"]


def test_serve_t4r_with_torchscript(tmpdir):
    min_session_len = 5
    max_session_len = 20

    # ===========================================
    # Generate training data
    # ===========================================
    torch_yoochoose_like = tr.data.tabular_sequence_testing_data.torch_synthetic_data(
        num_rows=100,
        min_session_length=min_session_len,
        max_session_length=max_session_len,
        device="cuda",
    )

    # ===========================================
    # Translate T4R schema to Merlin schema
    # ===========================================

    # TODO: This should be a Merlin schema. For now, let's convert it across
    # in this test, but ultimately we should rework T4R to use Merlin Schemas.

    t4r_yoochoose_schema = t4r.data.tabular_sequence_testing_data.schema

    merlin_yoochoose_schema = Schema()
    for column in t4r_yoochoose_schema:
        name = column.name

        # The feature types in the T4R schemas are a bit hard to work with:
        # https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/538fc54bb8f2e3dc79224e497bebee15b00e4ab7/merlin_standard_lib/proto/schema_bp.py#L43-L53
        dtype = {0: np.float32, 2: np.int64, 3: np.float32}[column.type]
        tags = column.tags
        is_list = column.value_count.max > 0
        value_count = (
            {"min": max_session_len, "max": max_session_len} if is_list else {"min": 1, "max": 1}
        )
        is_ragged = is_list and value_count.get("min", 0) != value_count.get("max", 0)
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

    # Check that the translated schema types match the actual types of the values
    non_matching_dtypes = {}
    for key, value in torch_yoochoose_like.items():
        dtypes = (value.cpu().numpy().dtype, merlin_yoochoose_schema[key].dtype)
        if dtypes[0] != dtypes[1]:
            non_matching_dtypes[key] = dtypes

    assert len(non_matching_dtypes) == 0

    # ===========================================
    # Build, train, test, and JIT the model
    # ===========================================

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
    model = model.cuda()

    _ = model(torch_yoochoose_like, training=False)

    model.eval()

    adapted_model = ServingAdapter(model)

    traced_model = torch.jit.trace(adapted_model, torch_yoochoose_like, strict=True)
    assert isinstance(traced_model, torch.jit.TopLevelTracedModule)
    assert torch.allclose(
        model(torch_yoochoose_like)["predictions"],
        traced_model(torch_yoochoose_like),
    )

    # ===========================================
    # Build a simple Ensemble graph
    # ===========================================

    output_schema = Schema([ColumnSchema("output", dtype=np.float32)])

    torch_op = merlin_yoochoose_schema.column_names >> PredictPyTorch(
        traced_model, merlin_yoochoose_schema, output_schema
    )

    ensemble = Ensemble(torch_op, merlin_yoochoose_schema)
    ens_config, node_configs = ensemble.export(str(tmpdir))

    # ===========================================
    # Convert training data to Triton format
    # ===========================================

    df_cols = {}
    for name, tensor in torch_yoochoose_like.items():
        df_cols[name] = tensor.cpu().numpy().astype(merlin_yoochoose_schema[name].dtype)
        if len(tensor.shape) > 1:
            df_cols[name] = list(df_cols[name])

    df = make_df(df_cols)[merlin_yoochoose_schema.column_names]

    inputs = convert_df_to_triton_input(merlin_yoochoose_schema, df.iloc[:3])

    # ===========================================
    # Send request to Triton and check response
    # ===========================================

    outputs = [grpcclient.InferRequestedOutput(col) for col in output_schema.column_names]

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer("ensemble_model", inputs, outputs=outputs)

    assert response
