#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2022 NVIDIA Corporation. All Rights Reserved.
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
# ==============================================================================


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_systems_serving-ranking-models-with-merlin-systems/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Serving Ranking Models With Merlin Systems
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container. This Jupyter notebook example demonstrates how to deploy a ranking model to Triton Inference Server (TIS) and generate prediction results for a given query. As a prerequisite, the ranking model must be trained and saved with Merlin Models. Please read the [README](https://github.com/NVIDIA-Merlin/systems/blob/main/examples/README.md) for the instructions.
# 
# ## Overview
# 
# NVIDIA Merlin is an open source framework that accelerates and scales end-to-end recommender system pipelines. The Merlin framework is broken up into several sub components, these include: Merlin-Core, Merlin-Models, NVTabular and Merlin-Systems. Merlin Systems will be the focus of this example.
# 
# The purpose of the Merlin Systems library is to make it easy for Merlin users to quickly deploy their recommender systems from development to [Triton Inference Server](https://github.com/triton-inference-server/server). We extended the same user-friendly API users are accustomed to in NVTabular and leveraged it to accommodate deploying recommender system components to TIS. 
# 
# There are some points we need ensure before we continue with this Notebook. Please ensure you have a working NVTabular workflow and model stored in an accessible location. Merlin Systems take the data preprocessing workflow defined in NVTabular and load that into Triton Inference Server as a model. Subsequently it does the same for the trained model. Lets take a closer look at how Merlin Systems makes deploying to TIS simple and effortless, in the rest of this notebook.
# 
# ### Learning objectives
# 
# In this notebook, we learn how to deploy a NVTabular Workflow and a trained Tensorflow model from Merlin Models to Triton.
# - Load NVTabular Workflow
# - Load Pre-trained Merlin Models model
# - Create Ensemble Graph
# - Export Ensemble Graph
# - Run Tritonserver
# - Send Request to Tritonserver
# 
# ### Dataset
# 
# We use the synthetic train and test datasets generated by mimicking the real [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1) dataset to build our recommender system ranking models. To see how the data is transformed with NVTabular and how a DLRM model is trained with Merlin Models check out the [04-Exporting-ranking-models.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb) example notebook which is a prerequisite for this notebook.
# 
# It is important to note that the steps take in this notebook are generalized and can be applied to any set of workflow and models. 
# 
# ### Tools
# 
# - NVTabular
# - Merlin Models
# - Merlin Systems
# - Triton Inference Server

# ## Install Required Libraries
# 
# Install TensorFlow so we can read the saved model from disk.

# In[2]:


#!pip install tensorflow-gpu


# ## Load an NVTabular Workflow
# 
# First, we load the `nvtabular.Workflow` that we created in with this [example](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb). 

# In[3]:


import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
from nvtabular.workflow import Workflow

input_path = os.environ.get("INPUT_FOLDER", "/workspace/data/")

workflow_stored_path = os.path.join(input_path, "workflow")

workflow = Workflow.load(workflow_stored_path)


# After we load the workflow, we remove the label columns from it's inputs. This removes all columns with the `TARGET` tag from the workflow. We do this because we need to set the workflow to only  require the features needed to predict, not train, when creating an inference pipeline.

# In[4]:


from merlin.schema.tags import Tags

label_columns = workflow.output_schema.select_by_tag(Tags.TARGET).column_names
workflow.remove_inputs(label_columns)


# ## Load the Tensorflow Model
# 
# After loading the workflow, we load the model. This model was trained with the output of the workflow from the [Exporting Ranking Models](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb) example from Merlin Models.

# First, we need to import the Merlin Models library. Loading a TensorFlow model, which is based on custom subclasses, requires to the subclass definition. Otherwise, TensorFlow cannot load correctly load the model.

# In[5]:


import merlin.models.tf as mm


# In[6]:


import tensorflow as tf
tf_model_path = os.path.join(input_path, "dlrm")

model = tf.keras.models.load_model(tf_model_path)


# ## Create the Ensemble Graph
# 
# After we have both the model and the workflow loaded, we can create the ensemble graph. You create the graph. The goal is to illustrate the path of data through your full system. In this example we only serve a workflow with a model, but you can add other components that help you meet your business logic requirements.
# 
# Because this example has two components&mdash;a model and a workflow&mdash;we require two operators. These operators, also known as inference operators, are meant to abstract away all the "hard parts" of loading a specific component, such as a workflow or model, into Triton Inference Server. 
# 
# The following code block shows how to use two inference operators:
# 
# <dl>
#     <dt><code>TransformWorkflow</code></dt>
#     <dd>This operator ensures that the workflow is correctly saved and packaged with the required config so the server will know how to load it.</dd>
#     <dt><code>PredictTensorflow</code></dt>
#     <dd>This operator will do something similar with the model, loaded before.</dd>
# </dl>
# 
# Let's give it a try.

# In[7]:


from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.tensorflow import PredictTensorflow

serving_operators = workflow.input_schema.column_names >> TransformWorkflow(workflow) >> PredictTensorflow(model)


# ## Export Graph as Ensemble
# 
# The last step is to create the ensemble artifacts that Triton Inference Server can consume.
# To make these artifacts, we import the `Ensemble` class.
# The class is responsible for interpreting the graph and exporting the correct files for the server.
# 
# After you run the following cell, you'll see that we create a `ColumnSchema` for the expected inputs to the workflow.
# The workflow is a `Schema`. 
# 
# When you are creating an `Ensemble` object you supply the graph and a schema representing the starting input of the graph. the inputs to the ensemble graph are the inputs to the first operator of your graph. 
# 
# After you have created the `Ensemble` you export the graph, supplying an export path for the `Ensemble.export` function.
# 
# This returns an ensemble config which represents the entire inference pipeline and a list of node-specific configs.
# 
# Let's take a look below.

# In[8]:


workflow.output_schema


# In[ ]:


from merlin.systems.dag.ensemble import Ensemble
import numpy as np

ensemble = Ensemble(serving_operators, workflow.input_schema)

export_path = os.path.join(input_path, "ensemble")

ens_conf, node_confs = ensemble.export(export_path)


# Display the path to the directory with the ensemble.

# In[10]:


export_path


# ## Verification of Ensemble Artifacts
# 
# After we export the ensemble, we can check the export path for the graph's artifacts. The directory structure represents an ordering number followed by an operator identifier such as `1_transformworkflow`, `2_predicttensorflow`, and so on.
# 
# Inside each of those directories, the `export` method writes a `config.pbtxt` file and a directory with a number. The number indicates the version and begins at 1. The artifacts for each operator are found inside the `version` folder. These artifacts vary depending on the operator in use. 
# 
# Install the `seedir` python package so we can view some of the directory contents.

# In[11]:


# install seedir
get_ipython().system('pip install seedir')


# In[12]:


import seedir as sd

sd.seedir(export_path, style='lines', itemlimit=10, depthlimit=3, exclude_folders='.ipynb_checkpoints', sort=True)


# ## Starting Triton Inference Server
# 
# After we export the ensemble, we are ready to start the Triton Inference Server. The server is installed in all the Merlin inference containers.  If you are not using one of our containers, then ensure it is installed in your environment. For more information, see the Triton Inference Server [documentation](https://github.com/triton-inference-server/server/blob/r22.03/README.md#documentation). 
# 
# You can start the server by running the following command:
# 
# ```shell
# tritonserver --model-repository=/workspace/data/ensemble --backend-config=tensorflow,version=2
# ```
# 
# For the `--model-repository` argument, specify the same value as the `export_path` that you specified previously in the `ensemble.export` method.
# 
# After you run the `tritonserver` command, wait until your terminal shows messages like the following example:
# 
# ```shell
# I0414 18:29:50.741833 4067 grpc_server.cc:4421] Started GRPCInferenceService at 0.0.0.0:8001
# I0414 18:29:50.742197 4067 http_server.cc:3113] Started HTTPService at 0.0.0.0:8000
# I0414 18:29:50.783470 4067 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
# ```

# ## Retrieving Recommendations from Triton Inference Server
# 
# Now that our server is running, we can send requests to it. This request is composed of values that correspond to the request schema that was created when we exported the ensemble graph.
# 
# In the code below we create a request to send to triton and send it. We will then analyze the response, to show the full experience.
# 
# First we need to ensure that we have a client connected to the server that we started. To do this, we use the Triton HTTP client library.

# In[13]:


import tritonclient.http as client

# Create a triton client
try:
    triton_client = client.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# After we create the client and verified it is connected to the server instance, we can communicate with the server and ensure all the models are loaded correctly.

# In[14]:


# ensure triton is in a good state
triton_client.is_server_live()
triton_client.get_model_repository_index()


# After verifying the models are correctly loaded by the server, we use some original validation data and send it as an inference request to the server. Here the `valid` folder was generated after `04-Exporting-ranking-models.ipynb` notebook is run. 
# 
# > The `df_lib` object is `cudf` if a GPU is available and `pandas` otherwise.

# In[13]:


from merlin.core.dispatch import get_lib

df_lib = get_lib()

original_data_path = os.environ.get("INPUT_FOLDER", "/workspace/data/")

# read in data for request
batch = df_lib.read_parquet(
    os.path.join(original_data_path,"valid", "part.0.parquet"), num_rows=3, columns=workflow.input_schema.column_names
)
batch


# After we isolate our `batch`, we convert the dataframe representation into inputs for Triton. We also declare the outputs that we expect to receive from the model.

# In[16]:


from merlin.systems.triton import convert_df_to_triton_input
import tritonclient.grpc as grpcclient
# create inputs and outputs

inputs = convert_df_to_triton_input(workflow.input_schema.column_names, batch, grpcclient.InferInput)

output_cols = ensemble.graph.output_schema.column_names

outputs = [
    grpcclient.InferRequestedOutput(col)
    for col in output_cols
]


# Now that our `inputs` and `outputs` are created, we can use the `triton_client` that we created earlier to send the inference request.

# In[17]:


# send request to tritonserver
with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer("ensemble_model", inputs, request_id="1", outputs=outputs)


# When the server completes the inference request, it returns a response. This response is parsed to get the desired predictions.

# In[18]:


# access individual response columns to get values back.
for col in ensemble.graph.output_schema.column_names:
    print(col, response.as_numpy(col), response.as_numpy(col).shape)


# ## Summary
# 
# This sample notebook started with an exported DLRM model and workflow.  We saw how to create an ensemble graph, verify the ensemble artifacts in the file system, and then put the ensemble into production with Triton Inference Server.  Finally, we sent a simple inference request to the server and printed the response.
