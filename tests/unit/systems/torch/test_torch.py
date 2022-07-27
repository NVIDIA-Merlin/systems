import pathlib
from distutils.spawn import find_executable
from pathlib import Path

import numpy as np
import pytest
import torch
from google.protobuf import text_format  # noqa

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.pytorch import PredictPyTorch

# from merlin.systems.dag.ensemble import Ensemble
from tests.unit.systems.utils.triton import run_triton_server

TRITON_SERVER_PATH = find_executable("tritonserver")

triton = pytest.importorskip("merlin.systems.triton")
grpcclient = pytest.importorskip("tritonclient.grpc")
model_config_pb2 = pytest.importorskip("tritonclient.grpc.model_config_pb2")

model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

model_input_schema = Schema(
    [ColumnSchema("input", properties={"value_count": {"min": 3, "max": 3}}, dtype=np.float32)]
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

ptorch_op = pytest.importorskip("merlin.systems.dag.ops.pytorch")


def test_pytorch_op_exports_own_config(tmpdir):
    triton_op = ptorch_op.PredictPyTorch(model, model_input_schema, model_output_schema)

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
        assert parsed.backend == "pytorch"


def test_torch_backend(tmpdir):
    model_repository = Path(tmpdir)

    model_dir = model_repository / model_name
    model_version_dir = model_dir / "1"
    model_version_dir.mkdir(parents=True, exist_ok=True)

    # Write config out
    config_path = model_dir / "config.pbtxt"
    with open(str(config_path), "w") as f:
        f.write(model_config)

    # Write model
    model_scripted = torch.jit.script(model)
    model_scripted.save(str(model_version_dir / "model.pt"))

    input_data = np.array([[2.0, 3.0, 4.0], [4.0, 8.0, 1.0]]).astype(np.float32)

    inputs = [
        grpcclient.InferInput(
            "input", input_data.shape, triton.np_to_triton_dtype(input_data.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    assert response.as_numpy("OUTPUT__0").shape == (input_data.shape[0],)


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
def test_pytorch_op_serving(tmpdir):
    from merlin.core.dispatch import make_df
    from merlin.systems.dag.ensemble import Ensemble
    from tests.unit.systems.utils.triton import _run_ensemble_on_tritonserver

    model_name = "0_predictpytorch"
    model_path = str(tmpdir / "model.pt")

    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)

    predictions = ["input"] >> ptorch_op.PredictPyTorch(
        model_path, model_input_schema, model_output_schema
    )
    ensemble = Ensemble(predictions, model_input_schema)
    ens_config, node_configs = ensemble.export(tmpdir)

    input_data = np.array([[2.0, 3.0, 4.0], [4.0, 8.0, 1.0]]).astype(np.float32)

    inputs = [
        grpcclient.InferInput(
            "input", input_data.shape, triton.np_to_triton_dtype(input_data.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    assert response.as_numpy("OUTPUT__0").shape == (input_data.shape[0],)

    # TODO: This code fails in `make_df`, which prevents us from creating the right input data
    # TODO: Fix that somehow

    # input_data = np.array([[2.0, 3.0, 4.0]]).astype(np.float32)
    # request = make_df({"input": input_data})
    # response = _run_ensemble_on_tritonserver(
    #     tmpdir, ensemble.graph.output_schema.column_names, request, "0_predictpytorch"
    # )

    # assert response is not None
    # assert len(response.as_numpy("OUTPUT__0")) == 10
