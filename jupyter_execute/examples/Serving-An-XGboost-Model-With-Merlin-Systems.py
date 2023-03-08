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

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.


# <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_systems_serving-an-xgboost-model-with-merlin-systems/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Serving an XGBoost Model with Merlin Systems
# 
# This notebook is created using the latest stable [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) container. This Jupyter notebook example demonstrates how to deploy an `XGBoost` model to Triton Inference Server (TIS) and generate prediction results for a given query.
# 
# ## Overview
# 
# NVIDIA Merlin is an open source framework that accelerates and scales end-to-end recommender system pipelines. The Merlin framework is broken up into several sub components, these include: Merlin-Core, Merlin-Models, NVTabular and Merlin-Systems. Merlin Systems will be the focus of this example.
# 
# The purpose of the Merlin Systems library is to make it easy for Merlin users to quickly deploy their recommender systems from development to [Triton Inference Server](https://github.com/triton-inference-server/server). We extended the same user-friendly API users are accustomed to in NVTabular and leveraged it to accommodate deploying recommender system components to TIS. 
# 
# ### Learning objectives
# 
# In this notebook, we learn how to deploy a NVTabular Workflow and a trained XGBoost model from Merlin Models to Triton.
# - Create Ensemble Graph
# - Export Ensemble Graph
# - Run Triton server
# - Send request to Triton and verify results
# 
# ### Dataset
# 
# We use the [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/). It consists of ratings a user has given a movie along with some metadata for the user and the movie. We train an XGBoost model to predict the rating based on user and item features and proceed to deploy it to the Triton Inference Server.
# 
# It is important to note that the steps take in this notebook are generalized and can be applied to any set of workflow and models. 
# 
# ### Tools
# 
# - NVTabular
# - Merlin Models
# - Merlin Systems
# - Triton Inference Server

# ## Prerequisite: Preparing the data and Training XGBoost

# In this tutorial our objective is to demonstrate how to serve an `XGBoost` model. In order for us to be able to do so, we begin by downloading data and training a model. We breeze through these activities below.
# 
# If you would like to learn more about training an `XGBoost` model using the Merlin Models library, please consult this [tutorial](https://github.com/NVIDIA-Merlin/models/blob/main/examples/07-Train-an-xgboost-model-using-the-Merlin-Models-API.ipynb).

# In[ ]:


from merlin.core.utils import Distributed
from merlin.models.xgb import XGBoost
import nvtabular as nvt
import numpy as np
from merlin.schema.tags import Tags

from merlin.datasets.entertainment import get_movielens


# In[ ]:


ensemble_export_path = os.environ.get("OUTPUT_DATA_DIR", "ensemble")


# In[3]:


train, _ = get_movielens(variant='ml-100k')

preprocess_categories = ['movieId', 'userId', 'genres'] >> nvt.ops.Categorify(freq_threshold=2, dtype=np.int32)
preprocess_rating = ['rating'] >> nvt.ops.AddTags(tags=[Tags.TARGET, Tags.REGRESSION])

train_workflow = nvt.Workflow(preprocess_categories + preprocess_rating + train.schema.without(['rating_binary', 'title']).column_names)
train_transformed = train_workflow.fit_transform(train)

with Distributed():
    model = XGBoost(schema=train_transformed.schema)
    model.fit(
        train_transformed,
        num_boost_round=85,
        verbose_eval=20
)


# ## Create the Ensemble Graph

# Let us now define an `Ensemble` that will be used for serving predictions on the Triton Inference Server.
# 
# An `Ensemble` defines operations to be performed on incoming requests. It begins with specifying what fields that the inference request will contain.
# 
# Our model was trained on data that included the `target` column. However, in production, this information will not be available to us.
# 
# In general, you want to define a preprocessing workflow once and apply it throughout the lifecycle of your model, from training all the way to serving in production. Redefining the workflows on the go, or using custom written code for these operations, can be a source of subtle bugs.
# 
# In order to ensure we process our data in the same way in production as we do in training, let us now modify the training preprocessing pipeline and use it to construct our inference workflow.

# In[4]:


inf_workflow = train_workflow.remove_inputs(['rating'])


# Equipped with the modified data preprocessing workflow, let us define the full set of inference operations we will want to run on the Triton Inference Server.
# 
# We begin by stating what data the server can expect (`inf_workflow.input_schema.column_names`). We proceed to wrap our `inf_workflow` in `TransformWorkflow` -- an operator we can leverage for executing our NVTabular workflow during serving.
# 
# Last but not least, having received and preprocessed the data, we instruct the Triton Inference Server to perform inference using the model that we trained. 

# In[5]:


from merlin.systems.dag.ops.fil import PredictForest
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.workflow import TransformWorkflow

inf_ops = inf_workflow.input_schema.column_names >> TransformWorkflow(inf_workflow) \
                    >> PredictForest(model, inf_workflow.output_schema)


# With inference operations defined, all that remains now is outputting the ensemble to disk so that it can be loaded up when Triton starts.

# In[6]:


ensemble = Ensemble(inf_ops, inf_workflow.input_schema)
ensemble.export(ensemble_export_path);


# ## Starting the Triton Inference Server

# After we export the ensemble, we are ready to start the Triton Inference Server. The server is installed in Merlin Tensorflow and Merlin PyTorch containers. If you are not using one of our containers, then ensure it is installed in your environment. For more information, see the Triton Inference Server [documentation](https://github.com/triton-inference-server/server/blob/r22.03/README.md#documentation).
# 
# You can start the server by running the following command:
# 
# ```shell
# tritonserver --model-repository=ensemble
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
# In the code below we create a request to send to Triton and send it. We will then analyze the response, to show the full experience.
# 
# We begin by obtaining 10 examples from our train data to include in the request.

# In[7]:


ten_examples = train.compute()
ten_examples.head()


# Let's now package the information up as inputs and send it to Triton for inference.

# In[8]:


from merlin.systems.triton import convert_df_to_triton_input
import tritonclient.grpc as grpcclient

ten_examples = train.compute().drop(columns=['rating', 'title', 'rating_binary'])[:10]
inputs = convert_df_to_triton_input(inf_workflow.input_schema, ten_examples, grpcclient.InferInput)

outputs = [
    grpcclient.InferRequestedOutput(col)
    for col in inf_ops.output_schema.column_names
]
# send request to tritonserver
with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer("executor_model", inputs, outputs=outputs)


# We can now compare the predictions from the server to those from our local model.

# In[9]:


predictions_from_triton = response.as_numpy(outputs[0].name())


# In[10]:


with Distributed():
    local_predictions = model.predict(train_transformed)[:10]


# In[11]:


assert np.allclose(predictions_from_triton, local_predictions)


# We managed to preprocess the data in the same way in serving as we did during training and obtain the same predictions!
