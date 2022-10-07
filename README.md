# [Merlin Systems](https://github.com/NVIDIA-Merlin/systems)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/merlin-systems)
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-systems.svg)](https://pypi.python.org/pypi/merlin-systems/)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/systems)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/systems/main/README.html)

Merlin Systems provides tools for combining recommendation models with other elements of production recommender systems like feature stores, nearest neighbor search, and exploration strategies into end-to-end recommendation pipelines that can be served with [Triton Inference Server](https://github.com/triton-inference-server/server).

## Quickstart

Merlin Systems uses the Merlin Operator DAG API, the same API used in [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) for feature engineering, to create serving ensembles. To combine a feature engineering workflow and a Tensorflow model into an inference pipeline:

```python
import tensorflow as tf
from nvtabular.workflow import Workflow
from merlin.systems.dag import Ensemble, PredictTensorflow, TransformWorkflow

# Load saved NVTabular workflow and TensorFlow model
workflow = Workflow.load(nvtabular_workflow_path)
model = tf.keras.models.load_model(tf_model_path)

# Remove target/label columns from feature processing workflowk
workflow = workflow.remove_inputs([<target_columns>])

# Define ensemble pipeline
pipeline = (
	workflow.input_schema.column_names >>
	TransformWorkflow(workflow) >>
	PredictTensorflow(model)
)

# Export artifacts to disk
ensemble = Ensemble(pipeline, workflow.input_schema)
ensemble.export(export_path)
```

After you export your ensemble, you reference the directory to run an instance of Triton Inference Server to host your ensemble.

```shell
tritonserver --model-repository=/export_path/
```

Refer to the [Merlin Systems Example Notebooks](./examples/) for a notebook that serves a ranking models ensemble.
The notebook shows how to deploy the ensemble and demonstrates sending requests to Triton Inference Server.

## Building a Four-Stage Recommender Pipeline

Merlin Systems can also build more complex serving pipelines that integrate multiple models and external tools (like feature stores and nearest neighbor search):

```python
# Load artifacts for the pipeline
retrieval_model = tf.keras.models.load_model(retrieval_model_path)
ranking_model = tf.keras.models.load_model(ranking_model_path)
feature_store = feast.FeatureStore(feast_repo_path)

# Define the fields expected in requests
request_schema = Schema([
    ColumnSchema("user_id", dtype=np.int32),
])

# Fetch user features, use them to a compute user vector with retrieval model,
# and find candidate items closest to the user vector with nearest neighbor search
user_features = request_schema.column_names >> QueryFeast.from_feature_view(
    store=feature_store, view="user_features", column="user_id"
)

retrieval = (
    user_features
    >> PredictTensorflow(retrieval_model_path)
    >> QueryFaiss(faiss_index_path, topk=100)
)

# Filter out candidate items that have already interacted with
# in the current session and fetch item features for the rest
filtering = retrieval["candidate_ids"] >> FilterCandidates(
    filter_out=user_features["movie_ids"]
)

item_features = filtering >> QueryFeast.from_feature_view(
    store=feature_store, view="movie_features", column="filtered_ids",
)

# Join user and item features for the candidates and use them to predict relevance scores
combined_features = item_features >> UnrollFeatures(
    "movie_id", user_features, unrolled_prefix="user"
)

ranking = combined_features >> PredictTensorflow(ranking_model_path)

# Sort candidate items by relevance score with some randomized exploration
ordering = combined_features["movie_id"] >> SoftmaxSampling(
    relevance_col=ranking["output"], topk=10, temperature=20.0
)

# Create and export the ensemble
ensemble = Ensemble(ordering, request_schema)
ensemble.export("./ensemble")
```

## Installation

Merlin Systems requires Triton Inference Server and Tensorflow. The simplest setup is to use the [Merlin Tensorflow Inference Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference), which has both pre-installed.

### Installing Merlin Systems Using Pip

You can install Merlin Systems with `pip`:

```shell
pip install merlin-systems
```

### Installing Merlin Systems from Source

Merlin Systems can be installed from source by cloning the GitHub repository and running `setup.py`

```shell
git clone https://github.com/NVIDIA-Merlin/systems.git
cd systems && python setup.py develop
```

### Running Merlin Systems from Docker

Merlin Systems is installed on multiple Docker containers that are available from the NVIDIA GPU Cloud (NGC) catalog.
The following table lists the containers that include Triton Inference Server for use with Merlin.

| Container Name      | Container Location                                                                     | Functionality                                                                      |
| --------------------| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `merlin-hugectr`    | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr>    | Merlin frameworks, HugeCTR, and Triton Inference Server                            |
| `merlin-tensorflow` | <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow> | Merlin frameworks selected for only Tensorflow support and Triton Inference Server |

If you want to add support for GPU-accelerated workflows, you will first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

## Feedback and Support

To report bugs or get help, please [open an issue](https://github.com/NVIDIA-Merlin/NVTabular/issues/new/choose).
