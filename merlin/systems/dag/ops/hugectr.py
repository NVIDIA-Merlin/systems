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
from typing import List, Optional, Union

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceOperator


class HugeCTR(InferenceOperator):
    """This operator takes a HugeCTR model and packages it correctly for tritonserver
    to run, on the hugectr backend.
    """

    def __init__(
        self,
        model,
        *,
        device_list: Optional[List[int]] = None,
        max_batch_size: int = 64,
        gpucache: Optional[bool] = None,
        hit_rate_threshold: Optional[float] = None,
        gpucacheper: Optional[float] = None,
        use_mixed_precision: Optional[bool] = None,
        scaler: Optional[float] = None,
        use_algorithm_search: Optional[bool] = None,
        use_cuda_graph: Optional[bool] = None,
        num_of_worker_buffer_in_pool: Optional[int] = None,
        num_of_refresher_buffer_in_pool: Optional[int] = None,
        cache_refresh_percentage_per_iteration: Optional[float] = None,
        default_value_for_each_table: float = 0.0,
        refresh_delay: Optional[float] = None,
        refresh_interval: Optional[float] = None,
        freeze_sparse: Optional[bool] = None,
        max_nnz: Optional[int] = None,
        embeddingkey_long_type: Optional[bool] = None,
        supportlonglong: Optional[bool] = None,
        persistent_db: Optional[dict] = None,
        volatile_db: Optional[dict] = None,
        update_source: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        model : hugectr.Model, required
            A hugeCTR model instance.
        device_list : List[int]
            Indicate the list of devices used to deploy the
            Hierarchical Parameter Server (HPS). The default is an
            empty list.
        max_batch_size : int
            The maximum batch size to be processed per batch, by an
            inference request
        gpucache : bool
            Use this option to enable the GPU embedding cache mechanism.
        hit_rate_threshold : float
            Determines the insertion mechanism of the embedding cache
            and Parameter Server based on the hit rate.
        gpucacheper : float
            Determines what percentage of the embedding vectors will
            be loaded from the embedding table into the GPU embedding
            cache.
        use_mixed_precision: bool
            Determines if mixed precision will be used.
        scaler : float
            Scaler for parameter server model config.
        use_algorithm_search : bool
            Determines if algorithm search will be used.
        use_cuda_graph : bool
            Determines if cuda graph will be used.
        num_of_worker_buffer_in_pool : int
            Specifies number of worker buffers in pool.
        num_of_refresher_buffer_in_pool : int
            Specifies number of refresher buffers in pool.
        cache_refresh_percentage_per_iteration : float
            The percentage of the cache to refresh each iteration.
        default_value_for_each_table : float
            The default value to  use for each embedding table.
        refresh_delay : float
            Model refresh delay
        refresh_interval : float
            Model refresh interval
        freeze_sparse : bool
            Option to keep sparse tables from being updated.
            This is useful when using online updates if you wish
            to disable repeaded updates to these embedding tables.
        max_nnz : int
            Maximum NNZ
        supportlonglong : bool
            Parameter server config. Specifies if longlong is supported.
        persistent_db : dict, optional
            Configuration for persistent database.
            Supports RocsDB.
        volatile_db : dict, optional
            configuration for Volatile database. Allows utilizing
            Redis cluster deployments, to store and retrieve
            embeddings in/from the RAM memory available in your
            cluster.
        update_source : dict, optional
            Configuration of real-time update source for model
            updates. Supports Apache Kafka.
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.device_list = device_list or []
        self.hit_rate_threshold = hit_rate_threshold
        self.gpucache = gpucache
        self.gpucacheper = gpucacheper
        self.use_mixed_precision = use_mixed_precision
        self.scaler = scaler
        self.use_algorithm_search = use_algorithm_search
        self.use_cuda_graph = use_cuda_graph
        self.num_of_worker_buffer_in_pool = num_of_worker_buffer_in_pool
        self.num_of_refresher_buffer_in_pool = num_of_refresher_buffer_in_pool
        self.cache_refresh_percentage_per_iteration = cache_refresh_percentage_per_iteration
        self.default_value_for_each_table = default_value_for_each_table
        self.refresh_delay = refresh_delay
        self.refresh_interval = refresh_interval
        self.supportlonglong = supportlonglong
        self.persistent_db = persistent_db
        self.volatile_db = volatile_db
        self.update_source = update_source

        # These params will be set as parameters in the triton model config.
        self.model_config_params = dict(
            freeze_sparse=freeze_sparse,
            max_nnz=max_nnz,
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
        """Return the input schema for this operator.

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

    def export(self, path, input_schema, output_schema, node_id=None, params=None, version=1):
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
        params : string, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None.
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
        model_path = pathlib.Path(node_export_path) / str(version)
        model_path.mkdir(exist_ok=True)
        model_name = node_name

        # Write model files
        network_file = os.path.join(model_path, f"{model_name}.json")
        self.model.graph_to_json(graph_config_file=network_file)
        self.model.save_params_to_files(str(model_path) + "/")

        # Write parameter server configuration
        # TODO: support multiple models in same ensemble.
        # parameter server config will need to be centralized and
        # combine the models from more than one operator.
        model = self._get_ps_model_config(model_path, model_name)
        parameter_server_config = {
            "models": [model],
            "supportlonglong": self.supportlonglong,
        }
        if self.persistent_db:
            parameter_server_config["peristent_db"] = self.persistent_db
        if self.volatile_db:
            parameter_server_config["volatile_db"] = self.volatile_db
        if self.update_source:
            parameter_server_config["update_source"] = self.update_source
        parameter_server_config_path = str(node_export_path.parent / "ps.json")
        with open(parameter_server_config_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(parameter_server_config))

        # Write triton model config
        model_config_params = {**self.model_config_params, "network_file": network_file}
        config = self._get_model_config(node_name, model_config_params)
        with open(os.path.join(node_export_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        return config

    def _get_ps_model_config(self, model_path: Union[str, os.PathLike], model_name: str):
        """Get HugeCTR model config for parameter server.

        Parameters
        ----------
        model_path : str
            directory containing the exported model files.
        model_name : str
            The name of the model. A file of the name
            <model_name>.json is expected to be located in the model
            path provided.
        """
        model_path = pathlib.Path(model_path)

        network_file = model_path / f"{model_name}.json"

        # find paths to dense and sparse models
        dense_pattern = "*_dense_*.model"
        dense_path = [
            str(model_path / path.name)
            for path in model_path.glob(dense_pattern)
            if "opt" not in path.name
        ][0]
        sparse_pattern = "*_sparse_*.model"
        sparse_paths = [
            str(model_path / path.name)
            for path in model_path.glob(sparse_pattern)
            if "opt" not in path.name
        ]

        # find layers in model network file
        with open(network_file, "r", encoding="utf-8") as f:
            model_json = json.loads(f.read())
        data_layer = model_json["layers"][0]
        sparse_layers = [
            layer
            for layer in model_json["layers"]
            if layer["type"] == "DistributedSlotSparseEmbeddingHash"
        ]

        model = {}
        model["model"] = model_name
        model["network_file"] = network_file
        model["max_batch_size"] = self.max_batch_size
        model["dense_file"] = dense_path
        model["sparse_files"] = sparse_paths
        model["gpucache"] = self.gpucache
        model["hit_rate_threshold"] = self.hit_rate_threshold
        model["gpucacheper"] = self.gpucacheper
        model["use_mixed_precision"] = self.use_mixed_precision
        model["scaler"] = self.scaler
        model["use_algorithm_search"] = self.use_algorithm_search
        model["use_cuda_graph"] = self.use_cuda_graph
        model["num_of_worker_buffer_in_pool"] = self.num_of_worker_buffer_in_pool
        model["num_of_refresher_buffer_in_pool"] = self.num_of_refresher_buffer_in_pool
        model[
            "cache_refresh_percentage_per_iteration"
        ] = self.cache_refresh_percentage_per_iteration
        model["deployed_device_list"] = self.device_list
        model["default_value_for_each_table"] = [self.default_value_for_each_table] * len(
            sparse_layers
        )
        # each sample may contain a varying number of numeric (dense)
        # features.  this configures the value of the maximum number
        # of dense features in each sample, which determines the
        # pre-allocated memory size on the host and device.
        model["maxnum_des_feature_per_sample"] = data_layer["dense"]["dense_dim"]
        model["refresh_delay"] = self.refresh_delay
        model["refresh_interval"] = self.refresh_interval
        # This determines the pre-allocated memory size on the host and device.
        # We assume that for each input sample, there is a maximum
        # number of embedding keys per sample in each embedding table
        # that need to be looked up, so the user needs to configure
        # the [ Maximum(the number of embedding keys that need to be
        # queried from embedding table 1 in each sample), Maximum(the
        # number of embedding keys that need to be queried from
        # embedding table 2 in each sample), ...] in this item.
        model["maxnum_catfeature_query_per_table_per_sample"] = [
            len(x["sparse_embedding_hparam"]["slot_size_array"]) for x in sparse_layers
        ]
        model["embedding_vecsize_per_table"] = [
            x["sparse_embedding_hparam"]["embedding_vec_size"] for x in sparse_layers
        ]
        model["embedding_table_names"] = [x["top"] for x in sparse_layers]
        model["label_dim"] = data_layer["label"]["label_dim"]
        model["slot_num"] = sum(x["slot_num"] for x in data_layer["sparse"])

        # remove unset (None) values
        model = {k: v for k, v in model.items() if v is not None}

        return model

    def _get_model_config(self, name: str, parameters: dict) -> model_config.ModelConfig:
        """Returns a ModelConfig for a HugeCTR model.

        Parameters
        ----------
        name : string
            The name of the triton model. This should match the name
            of the directory where the model is exported.
        parameters : dict
            Dictionary holding parameter values for the model configuration.

        Returns
        -------
        config
            Dictionary representation of hugectr config.
        """
        config = model_config.ModelConfig(
            name=name,
            backend="hugectr",
            max_batch_size=self.max_batch_size,
            input=[
                model_config.ModelInput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1]),
                model_config.ModelInput(
                    name="CATCOLUMN", data_type=model_config.TYPE_INT64, dims=[-1]
                ),
                model_config.ModelInput(
                    name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1]
                ),
            ],
            output=[
                model_config.ModelOutput(
                    name="OUTPUT0", data_type=model_config.TYPE_FP32, dims=[-1]
                )
            ],
            instance_group=[
                model_config.ModelInstanceGroup(
                    gpus=self.device_list,
                    count=len(self.device_list),
                    kind=model_config.ModelInstanceGroup.Kind.KIND_GPU,
                )
            ],
        )

        for parameter_key, parameter_value in parameters.items():
            if parameter_value is None:
                continue
            if isinstance(parameter_value, list):
                config.parameters[parameter_key].string_value = json.dumps(parameter_value)
            elif isinstance(parameter_value, bool):
                config.parameters[parameter_key].string_value = str(parameter_value).lower()
            config.parameters[parameter_key].string_value = str(parameter_value)

        return config
