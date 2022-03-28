# Merlin Systems

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/merlin-systems)
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-systems.svg)](https://pypi.python.org/pypi/merlin-systems/)
[![Stability Alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://img.shields.io/badge/stability-alpha-f4d03f.svg)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/NVIDIA-Merlin/systems/CPU%20CI)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/systems)

Merlin Systems provides tools for combining recommendation models with other elements of production recommender systems like feature stores, nearest neighbor search, and exploration strategies into end-to-end recommendation pipelines that can be served with [Triton Inference Server](https://github.com/triton-inference-server/server).

## Installation

Merlin Systems requires Triton Inference Server and Tensorflow. The simplest setup is to use the [Merlin Tensorflow Inference Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference), which has both pre-installed.

### Installing Merlin Systems Using Pip

You can install Merlin Systems with `pip`:

```sh
pip install merlin-systems
```

### Installing Merlin Systems from Source

Merlin Systems can be installed from source by cloning the GitHub repository and running `setup.py`

```bash
git clone https://github.com/NVIDIA-Merlin/systems.git
cd systems && python setup.py develop
```

### Running Merlin Systems from Docker

Merlin Systems is installed on multiple Docker containers, which are available in the [NVIDIA Merlin container repository](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=dateModifiedDESC&query=merlin):

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-inference           | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-inference           | Merlin frameworks and Triton Inference Server |
| merlin-tensorflow-inference            | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference            | Merlin frameworks selected for only Tensorflow support and Triton Inference Server                    |

If you want to add support for GPU-accelerated workflows, you'will first need to install  the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

## Getting Started with Merlin Systems
Merlin systems is a library designed to allow users to easily deploy data science pipelines to the Triton Inference Server[https://developer.nvidia.com/nvidia-triton-inference-server]. Merlin systems is designed to simplify the inference deployment scheme. To do this we leverage the merlin DAG, the same graph API used in NVTabular[https://github.com/NVIDIA-Merlin/NVTabular] for feature engineering. Below we will illustrate how easy it is to setup a triton inference server instance running a feature engineering workflow that feeds into a tensorflow model. 

```python
import tensorflow as tf
from nvtabular.workflow import Workflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble
from merlin.schema import Schema, ColumnSchema

# Load saved NVTabular workflow and TensorFlow model
workflow = Workflow.load(nvtabular_workflow_path)
model = tf.keras.models.load_model(tf_model_path)
```

First step is to load the previously saved workflow and model. We will then do some necessary modifications to these components.

```python
# Modify workflow and model
workflow = workflow.remove_inputs([<target_columns>])
for key, spec in model._saved_model_inputs_spec.items():
    model._saved_model_inputs_spec[key] = tf.TensorSpec(shape=spec.shape, dtype=spec.dtype, name=key)
```
Once the model and workflow are loaded, they can be introduced into an Ensemble graph. Below, we set the inputs of the graph as the inputs of the workflow. you expect your workflow to receive all columns except label columns as inference request inputs to your Triton Server instance. Then, using the merlin DAG api, set the operator pipeline, sequentially identifying the steps of our inference pipeline. In this example we have two steps, the feature engineering (`TransformWorkflow(workflow)`) followed by model prediction (`PredictTensorflow(model)`). As shown below:

```python
# Define ensemble pipeline
triton_chain = (
	workflow.input_schema.column_names >> 
	TransformWorkflow(workflow) >> 
	PredictTensorflow(model)
)
```
Once you have created your graph, defining all the steps of your Inference pipeline, the next step is encapsulate that graph with an `Ensemble`. The `Ensemble` is responsible for interpreting the graph supplied and the supplied input_schema. After creating the ensemble, you need to call `export` on your `Ensemble` object. This will create a directory hosting necessary artifacts and a series of config for each of the steps in the inference pipeline and one config for the entire ensemble. 

```python
# Export artificats to disk
ensemble = Ensemble(triton_chain, workflow.input_schema)
ensemble.export(export_path)

```

Once you have successfully exported your ensemble you can use the created directory to run an instance of tritonserver hosting your ensemble.
```shell
tritonserver --model-repository=/export_path/
```

After the tritonserver successfully loads all the models of your ensemble, you will be ready to send requests to your inference pipeline.  


## Notebook Examples

In our repository, a [Getting Started Example](./examples/Getting_Started/Getting-started-with-Merlin-Systems.ipynb) Jupyter notebook is provided, which deploys a recommender system pipeline using Merlin Systems to Triton Inference Server. It loads the the NVTabular workflow and TensorFlow model, creates an ensemble workflow and exports the ensemble to Triton Inference Server. Finally, an example request is sent to Triton Inference Server to test the deployed ensemble.


## Feedback and Support

To report bugs or get help, please [open an issue](https://github.com/NVIDIA-Merlin/NVTabular/issues/new/choose).
