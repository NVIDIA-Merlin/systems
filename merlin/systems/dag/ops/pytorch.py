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
import json
import os
import pathlib
from shutil import copyfile
from typing import Dict, Optional

import cloudpickle

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch  # noqa
import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.core.dispatch import is_string_dtype  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.ops.operator import InferenceOperator  # noqa
from merlin.systems.triton.export import _convert_dtype  # noqa


class PredictPyTorch(InferenceOperator):
    """
    This operator takes a pytorch model and packages it correctly for tritonserver
    to run, on the pytorch backend.
    """

    def __init__(
        self,
        model_or_path,
        torchscript: bool,
        input_schema: Schema,
        output_schema: Schema,
        sparse_max: Optional[Dict[str, int]] = None,
        use_fix_dtypes: bool = False,
    ):
        """
        Instantiate a PredictPyTorch inference operator.

        Parameters
        ----------
        model_or_path : PyTorch model or string
            This can be a pytorch model or a path to a pytorch model.
        torchscript : bool
            Indicates whether the model is jit-compiled. If True, we use the optimized `pytorch`
            backend for Triton Inference Server. If False, uses the python backend.
        input_schema : Schema
            Input schema for the pytorch model. This could be the output schema of the NVTabular
            workflow that produced your training data.
        output_schema : Schema
            Output schema for the pytorch model.
        """
        self.sparse_max = sparse_max or {}
        self.use_fix_dtypes = use_fix_dtypes

        if isinstance(model_or_path, (str, os.PathLike)):
            self.path = model_or_path
            self.model = torch.load(self.path)
        else:
            self.path = None
            self.model = model_or_path

        # TODO: figure out if we can infer input / output schemas from the pytorch model. Now we
        # just make them parameters.
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.torchscript = torchscript

        super().__init__()

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Use the input schema supplied during object creation.
        """
        return self.input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """
        Use the output schema supplied during object creation.
        """
        return self.output_schema

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create a directory inside supplied path based on our export name"""
        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        export_model_path = pathlib.Path(node_export_path) / str(version)
        export_model_path.mkdir(exist_ok=True)

        if self.path:
            copyfile(
                str(self.path),
                export_model_path / "model.pt",
            )
            # TODO: We still need to copy the TritonPythonModel
        else:
            if self.torchscript:
                self.model.save(export_model_path / "model.pt")
            else:
                # python backend requires copying model.py and some other files into the right
                # directory. Plus the model.
                torch.save(self.model, export_model_path / "model.pt")

                pt_model_path = os.path.join(export_model_path, "model.pth")
                torch.save(self.model.state_dict(), pt_model_path)

                pt_model_path = os.path.join(export_model_path, "model.pkl")
                with open(pt_model_path, "wb") as o:
                    cloudpickle.dump(self.model, o)

                copyfile(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "triton",
                        "models",
                        "pytorch_model.py",
                    ),
                    os.path.join(export_model_path, "model.py"),
                )
                # breakpoint()

        return self._export_model_config(
            node_name, node_export_path, self.sparse_max, self.use_fix_dtypes, version
        )

    def _export_model_config(
        self, name, output_path, sparse_max: Dict[str, int], use_fix_dtypes: bool, version: int = 1
    ):
        """Exports a PyTorch model for serving with Triton

        Parameters
        ----------
        name:
            The name of the triton model to export
        output_path:
            The path to write the exported model to
        """
        config = (
            self._export_torchscript_config(name, output_path)
            if self.torchscript
            else self._export_python_config(name, output_path, sparse_max, use_fix_dtypes, version)
        )

        return config

    def _export_python_config(
        self,
        name: str,
        output_path: str,
        sparse_max: Dict[str, int],
        use_fix_dtypes: bool,
        version: int = 1,
    ):
        """Exports a PyTorch model for serving with Triton

        Parameters
        ----------
        name:
            The name of the triton model to export
        output_path:
            The path to write the exported model to
        """
        config = model_config.ModelConfig(name=name, backend="python")

        for col_name, col_schema in self.input_schema.column_schemas.items():
            dim = sparse_max[col_name] if sparse_max and col_name in sparse_max.keys() else 1
            _add_model_param(col_schema, model_config.ModelInput, config.input, [-1, dim])

        *_, last_layer = self.model.parameters()
        dims = last_layer.shape[0]
        dtype = last_layer.dtype
        config.output.append(
            model_config.ModelOutput(
                name="OUTPUT__0", data_type=_convert_pytorch_dtype(dtype), dims=[-1, dims]
            )
        )

        if sparse_max:
            with open(
                os.path.join(output_path, str(version), "model_info.json"), "w", encoding="utf-8"
            ) as o:
                model_info = {}
                model_info["sparse_max"] = sparse_max
                model_info["use_fix_dtypes"] = use_fix_dtypes
                json.dump(model_info, o)

        with open(os.path.join(output_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)
        return config

    def _export_torchscript_config(self, name, output_path):
        """Exports a PyTorch model for serving with Triton

        Parameters
        ----------
        name:
            The name of the triton model to export
        output_path:
            The path to write the exported model to
        """
        config = model_config.ModelConfig(name=name)

        config.backend = "pytorch"
        config.platform = "pytorch_libtorch"
        config.parameters["INFERENCE_MODE"].string_value = "true"

        for col_name, col_schema in self.input_schema.column_schemas.items():
            dims = [-1, 1]

            if col_schema.is_list and not col_schema.is_ragged:
                value_count = col_schema.properties.get("value_count", None)
                if value_count and value_count["min"] == value_count["max"]:
                    dims = [-1, value_count["max"]]

            config.input.append(
                model_config.ModelInput(
                    name=col_name, data_type=_convert_dtype(col_schema.dtype), dims=dims
                )
            )

        for col_name, col_schema in self.output_schema.column_schemas.items():
            # this assumes the list columns are 1D tensors both for cats and conts
            config.output.append(
                model_config.ModelOutput(
                    name=col_name,
                    data_type=_convert_dtype(col_schema.dtype),
                    dims=[-1, 1],
                )
            )

        with open(os.path.join(output_path, "config.pbtxt"), "wb") as o:
            text_format.PrintMessage(config, o)
        return config


def _add_model_param(col_schema, paramclass, params, dims=None):
    dims = dims if dims is not None else [-1, 1]
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=dims,
            )
        )
        params.append(
            paramclass(
                name=col_schema.name + "__nnzs", data_type=model_config.TYPE_INT64, dims=dims
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )


def _convert_pytorch_dtype(dtype):
    """converts a dtype to the appropriate triton proto type"""

    dtypes = {
        torch.float64: model_config.TYPE_FP64,
        torch.float32: model_config.TYPE_FP32,
        torch.float16: model_config.TYPE_FP16,
        torch.int64: model_config.TYPE_INT64,
        torch.int32: model_config.TYPE_INT32,
        torch.int16: model_config.TYPE_INT16,
        torch.int8: model_config.TYPE_INT8,
        torch.uint8: model_config.TYPE_UINT8,
        torch.bool: model_config.TYPE_BOOL,
    }

    if is_string_dtype(dtype):
        return model_config.TYPE_STRING
    elif dtype in dtypes:
        return dtypes[dtype]
    else:
        raise ValueError(f"Can't convert dtype {dtype})")
