from pathlib import Path

import numpy as np
import pytest
import torch

# from merlin.systems.dag.ensemble import Ensemble  # noqa
from tests.unit.systems.utils.triton import run_triton_server  # noqa

triton = pytest.importorskip("merlin.systems.triton")
grpcclient = pytest.importorskip("tritonclient.grpc")


model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))


def test_torch(tmpdir):
    model_repository = Path(tmpdir)

    model_name = "example_model"
    config = """
name: "example_model"
input: [
 {
    name: "input"
    data_type: TYPE_FP32
    dims: [ -1, 3 ]
  }
]
output [
 {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }
]
parameters: {
key: "INFERENCE_MODE"
    value: {
    string_value: "true"
    }
}
backend: "pytorch"
    """
    model_dir = model_repository / model_name
    model_version_dir = model_dir / "1"
    model_version_dir.mkdir(parents=True, exist_ok=True)

    # Write config out
    config_path = model_dir / "config.pbtxt"
    with open(str(config_path), "w") as f:
        f.write(config)

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
