import pathlib
from distutils.spawn import find_executable  # pylint: disable=deprecated-module
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from google.protobuf import text_format  # noqa
from torch import Tensor

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ensemble import Ensemble
from tests.unit.systems.utils.triton import run_triton_server

TRITON_SERVER_PATH = find_executable("tritonserver")

triton = pytest.importorskip("merlin.systems.triton")
grpcclient = pytest.importorskip("tritonclient.grpc")
ptorch_op = pytest.importorskip("merlin.systems.dag.ops.pytorch")
model_config_pb2 = pytest.importorskip("tritonclient.grpc.model_config_pb2")


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, input_dict: Dict[str, Tensor]):
        linear_out = self.linear(input_dict["input"].to(self.linear.weight.device))

        return linear_out


model = CustomModel()
model_scripted = torch.jit.script(model)

model_input_schema = Schema(
    [
        ColumnSchema(
            "input",
            properties={"value_count": {"min": 3, "max": 3}},
            dtype=np.float32,
            is_list=True,
            is_ragged=False,
        )
    ]
)
model_output_schema = Schema([ColumnSchema("OUTPUT__0", dtype=np.float32)])


model_name = "example_model"
model_config = """
name: "example_model"
platform: "pytorch_libtorch"
input {
  name: "input"
  data_type: TYPE_FP32
  dims: -1
  dims: 3
}
output {
  name: "OUTPUT__0"
  data_type: TYPE_FP32
  dims: -1
  dims: 1
}
parameters {
  key: "INFERENCE_MODE"
  value {
    string_value: "true"
  }
}
backend: "pytorch"
"""


@pytest.mark.parametrize("torchscript", [True])
def test_pytorch_op_exports_own_config(tmpdir, torchscript):
    model_to_use = model_scripted if torchscript else model

    triton_op = ptorch_op.PredictPyTorch(model_to_use, model_input_schema, model_output_schema)

    triton_op.export(tmpdir, None, None)

    # Export creates directory
    export_path = pathlib.Path(tmpdir) / triton_op.export_name
    assert export_path.exists()

    # Export creates the config file
    config_path = export_path / "config.pbtxt"
    assert config_path.exists()

    # Read the config file back in from proto
    with open(config_path, "rb") as f:
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, model_config_pb2.ModelConfig())

        # The config file contents are correct
        assert parsed.name == triton_op.export_name
        assert parsed.backend == ("pytorch" if torchscript else "python")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_torch_backend(tmpdir):
    model_repository = Path(tmpdir)

    model_dir = model_repository / model_name
    model_version_dir = model_dir / "1"
    model_version_dir.mkdir(parents=True, exist_ok=True)

    # Write config out
    config_path = model_dir / "config.pbtxt"
    with open(str(config_path), "w", encoding="utf-8") as f:
        f.write(model_config)

    # Write model
    model_scripted = torch.jit.script(model)
    model_scripted.save(str(model_version_dir / "model.pt"))

    input_data = {"input": np.array([[2.0, 3.0, 4.0], [4.0, 8.0, 1.0]]).astype(np.float32)}

    inputs = [
        grpcclient.InferInput(
            "input", input_data["input"].shape, triton.np_to_triton_dtype(input_data["input"].dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data["input"])

    outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    assert response.as_numpy("OUTPUT__0").shape[0] == input_data["input"].shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("torchscript", [True])
@pytest.mark.parametrize("use_path", [True, False])
def test_pytorch_op_serving(tmpdir, use_path, torchscript):
    model_name = "0_predictpytorch"
    model_path = str(tmpdir / "model.pt")

    model_to_use = model_scripted if torchscript else model
    model_or_path = model_path if use_path else model_to_use

    if use_path:
        try:
            # jit-compiled version of a model
            model_to_use.save(model_path)
        except AttributeError:
            # non-jit-compiled version of a model
            torch.save(model_to_use, model_path)

    predictions = ["input"] >> ptorch_op.PredictPyTorch(
        model_or_path, model_input_schema, model_output_schema
    )
    ensemble = Ensemble(predictions, model_input_schema)
    ens_config, node_configs = ensemble.export(tmpdir)

    input_data = {"input": np.array([[2.0, 3.0, 4.0], [4.0, 8.0, 1.0]]).astype(np.float32)}

    inputs = [
        grpcclient.InferInput(
            "input", input_data["input"].shape, triton.np_to_triton_dtype(input_data["input"].dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data["input"])

    outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    assert response.as_numpy("OUTPUT__0").shape[0] == input_data["input"].shape[0]
