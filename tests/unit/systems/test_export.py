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

from distutils.spawn import find_executable

import pytest

from merlin.io import Dataset
from merlin.systems.workflow import get_embedding_sizes
from nvtabular import Workflow, ops

tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")  # noqa

triton = pytest.importorskip("merlin.systems.triton")
data_conversions = pytest.importorskip("merlin.systems.triton.conversions")
ensemble = pytest.importorskip("merlin.systems.triton.export")

torch = pytest.importorskip("torch")  # noqa

from merlin.systems.triton.export import export_pytorch_ensemble, export_tensorflow_ensemble  # noqa
from tests.unit.systems.inference_utils import (  # noqa
    _run_ensemble_on_tritonserver,
    create_pytorch_model,
    create_tf_model,
)

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

TRITON_SERVER_PATH = find_executable("tritonserver")
tf_utils.configure_tensorflow()


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("output_model", ["tensorflow"])
def test_export_run_ensemble_triton(tmpdir, engine, output_model, df):
    conts = ["x", "y", "id"] >> ops.FillMissing() >> ops.Normalize()
    cats = ["name-cat", "name-string"] >> ops.Categorify(cat_cache="host")
    workflow = Workflow(conts + cats)
    dataset = Dataset(df)
    workflow.fit(dataset)

    embed_shapes = get_embedding_sizes(workflow)
    cat_cols = list(embed_shapes.keys())

    if output_model == "tensorflow":
        tf_model = create_tf_model(cat_cols, [], embed_shapes)
        export_tensorflow_ensemble(tf_model, workflow, "test_name", tmpdir, [])
    elif output_model == "pytorch":
        torch_model = create_pytorch_model(cat_cols, [], embed_shapes)
        export_pytorch_ensemble(
            torch_model,
            workflow,
            {},
            "test_name",
            tmpdir,
            [],
        )

    # assert os.path.exists(os.path.join(repo, "config.pbtxt"))
    tri_df = df.iloc[:10]
    tri_df = tri_df[["x", "y", "id", "name-cat", "name-string"]]
    response = _run_ensemble_on_tritonserver(str(tmpdir), ["output"], tri_df, "test_name")
    assert response is not None
    assert len(response.as_numpy("output")) == 10
