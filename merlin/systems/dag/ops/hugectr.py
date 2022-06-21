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

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceOperator


class HugeCTR(InferenceOperator):
    """
    Creates an operator meant to house a HugeCTR model.
    Allows the model to run as part of a merlin graph operations for inference.
    """

    def __init__(
        self,
        model,
        max_batch_size=1024,
        device_list=None,
        hit_rate_threshold=None,
        gpucache=None,
        freeze_sparse=None,
        gpucacheper=None,
        label_dim=None,
        slots=None,
        cat_feature_num=None,
        des_feature_num=None,
        max_nnz=None,
        embedding_vector_size=None,
        embeddingkey_long_type=None,
    ):
        self.model = model
        self.max_batch_size = max_batch_size or 1024
        self.device_list = device_list

        # if isinstance(model_or_path,(str, os.PathLike)):
        #     self.path = model_or_path
        # elif isinstance(model_or_path, hugectr.Model):
        #     self.model = model_or_path
        # else:
        #     raise ValueError(
        #         "Unsupported type for model_or_path. "
        #         "Must be pathlike or hugectr.Model"
        #     )

        self.hugectr_params = dict(
            hit_rate_threshold=hit_rate_threshold,
            gpucache=gpucache,
            freeze_sparse=freeze_sparse,
            gpucacheper=gpucacheper,
            label_dim=label_dim,
            slots=slots,
            cat_feature_num=cat_feature_num,
            des_feature_num=des_feature_num,
            max_nnz=max_nnz,
            embedding_vector_size=embedding_vector_size,
            embeddingkey_long_type=embeddingkey_long_type,
        )

        super().__init__()

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ):
        """_summary_

        Parameters
        ----------
        root_schema : Schema
            The original schema to the graph.
        parents_schema : Schema
            A schema comprised of the output schemas of all parent nodes.
        deps_schema : Schema
            A concatenation of the output schemas of all dependency nodes.
        selector : ColumnSelector
            Sub selection of columns required to compute the input schema.

        Returns
        -------
        Schema
            A schema describing the inputs of the model.
        """
        return Schema(
            [
                ColumnSchema("DES", dtype=np.float32),
                ColumnSchema("CATCOLUMN", dtype=np.int64),
                ColumnSchema("ROWINDEX", dtype=np.int32),
            ]
        )

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ):
        """Return output schema of the model.

        Parameters
        ----------
        input_schema : Schema
            Schema representing inputs to the model
        col_selector : ColumnSelector
            list of columns to focus on from input schema
        prev_output_schema : Schema, optional
            The output schema of the previous node, by default None

        Returns
        -------
        Schema
            Schema describing the output of the model.
        """
        return Schema([ColumnSchema("OUTPUT0", dtype=np.float32)])

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create and export the required config files for the hugectr model.

        Parameters
        ----------
        path : current path of the model
            _description_
        input_schema : Schema
            Schema describing inputs to model
        output_schema : Schema
            Schema describing outputs of model
        node_id : int, optional
            The node's position in execution chain, by default None
        version : int, optional
            The version of the model, by default 1

        Returns
        -------
        config
            Dictionary representation of config file in memory.
        """
        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name
        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        hugectr_model_path = pathlib.Path(node_export_path) / str(version)
        hugectr_model_path.mkdir(exist_ok=True)
        self.model.graph_to_json(graph_config_file=str(hugectr_model_path / "model.json"))
        self.model.save_params_to_files(str(hugectr_model_path) + "/")
        # generate config
        # save artifacts to model repository (path)
        # {node_id}_hugectr/config.pbtxt
        # {node_id}_hugectr/1/
        model_name = "model"
        dense_pattern = "*_dense_*.model"
        dense_path = [
            os.path.join(hugectr_model_path, path.name)
            for path in hugectr_model_path.glob(dense_pattern)
        ][0]
        sparse_pattern = "*_sparse_*.model"
        sparse_paths = [
            os.path.join(hugectr_model_path, path.name)
            for path in hugectr_model_path.glob(sparse_pattern)
        ]
        network_file = os.path.join(hugectr_model_path, f"{model_name}.json")

        config_dict = dict()
        config_dict["supportlonglong"] = True
        model = dict()
        model["model"] = model_name
        model["sparse_files"] = sparse_paths
        model["dense_file"] = dense_path
        model["network_file"] = network_file
        model["num_of_worker_buffer_in_pool"] = 4
        model["num_of_refresher_buffer_in_pool"] = 1
        model["deployed_device_list"] = self.device_list
        model["max_batch_size"] = (self.max_batch_size,)
        model["default_value_for_each_table"] = [0.0]
        model["hit_rate_threshold"] = 0.9
        model["gpucacheper"] = 0.5
        model["gpucache"] = True
        model["cache_refresh_percentage_per_iteration"] = 0.2
        config_dict["models"] = [model]

        parameter_server_config_path = str(hugectr_model_path / "ps.json")
        with open(parameter_server_config_path, "w") as f:
            f.write(json.dumps(config_dict))

        self.hugectr_params["config"] = parameter_server_config_path
        config = _hugectr_config(node_name, self.hugectr_params, max_batch_size=self.max_batch_size)

        with open(os.path.join(node_export_path, "config.pbtxt"), "w") as o:
            text_format.PrintMessage(config, o)

        return config


def _hugectr_config(name, hugectr_params, max_batch_size=None):
    """Create a config for a HugeCTR model.

    Parameters
    ----------
    name : string
        The name of the hugectr model.
    hugectr_params : dictionary
        Dictionary holding parameter values required by hugectr
    max_batch_size : int, optional
        The maximum batch size to be processed per batch, by an inference request, by default None

    Returns
    -------
    config
        Dictionary representation of hugectr config.
    """
    config = model_config.ModelConfig(name=name, backend="hugectr", max_batch_size=max_batch_size)

    config.input.append(
        model_config.ModelInput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1])
    )

    config.input.append(
        model_config.ModelInput(name="CATCOLUMN", data_type=model_config.TYPE_INT64, dims=[-1])
    )

    config.input.append(
        model_config.ModelInput(name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1])
    )

    config.output.append(
        model_config.ModelOutput(name="OUTPUT0", data_type=model_config.TYPE_FP32, dims=[-1])
    )

    config.instance_group.append(model_config.ModelInstanceGroup(gpus=[0], count=1, kind=1))

    config_hugectr = model_config.ModelParameter(string_value=hugectr_params["config"])
    config.parameters["config"].CopyFrom(config_hugectr)

    gpucache_val = hugectr_params.get("gpucache", "true")

    gpucache = model_config.ModelParameter(string_value=gpucache_val)
    config.parameters["gpucache"].CopyFrom(gpucache)

    gpucacheper_val = str(hugectr_params.get("gpucacheper_val", "0.5"))

    gpucacheper = model_config.ModelParameter(string_value=gpucacheper_val)
    config.parameters["gpucacheper"].CopyFrom(gpucacheper)

    label_dim = model_config.ModelParameter(string_value=str(hugectr_params["label_dim"]))
    config.parameters["label_dim"].CopyFrom(label_dim)

    slots = model_config.ModelParameter(string_value=str(hugectr_params["slots"]))
    config.parameters["slots"].CopyFrom(slots)

    des_feature_num = model_config.ModelParameter(
        string_value=str(hugectr_params["des_feature_num"])
    )
    config.parameters["des_feature_num"].CopyFrom(des_feature_num)

    cat_feature_num = model_config.ModelParameter(
        string_value=str(hugectr_params["cat_feature_num"])
    )
    config.parameters["cat_feature_num"].CopyFrom(cat_feature_num)

    max_nnz = model_config.ModelParameter(string_value=str(hugectr_params["max_nnz"]))
    config.parameters["max_nnz"].CopyFrom(max_nnz)

    embedding_vector_size = model_config.ModelParameter(
        string_value=str(hugectr_params["embedding_vector_size"])
    )
    config.parameters["embedding_vector_size"].CopyFrom(embedding_vector_size)

    embeddingkey_long_type_val = hugectr_params.get("embeddingkey_long_type", "true")

    embeddingkey_long_type = model_config.ModelParameter(string_value=embeddingkey_long_type_val)
    config.parameters["embeddingkey_long_type"].CopyFrom(embeddingkey_long_type)

    return config
