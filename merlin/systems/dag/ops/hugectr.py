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

from merlin.core.dispatch import make_df
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.schema.tags import Tags
from merlin.systems.dag.ops.compat import pb_utils
from merlin.systems.dag.ops.operator import (
    InferenceDataFrame,
    InferenceOperator,
    PipelineableInferenceOperator,
)


def _convert(data, slot_size_array, categorical_columns, labels=None):
    """Prepares data for a request to the HugeCTR predict interface.

    Returns
    -------
        Tuple of dense, categorical, and row index.
        Corresponding to the three inputs required by a HugeCTR model.
    """
    labels = labels or []
    dense_columns = list(set(data.columns) - set(categorical_columns + labels))
    categorical_dim = len(categorical_columns)
    batch_size = data.shape[0]

    shift = np.insert(np.cumsum(slot_size_array), 0, 0)[:-1].tolist()

    # These dtypes are static for HugeCTR
    dense = np.array([data[dense_columns].values.flatten().tolist()], dtype="float32")
    cat = np.array([(data[categorical_columns] + shift).values.flatten().tolist()], dtype="int64")
    rowptr = np.array([list(range(batch_size * categorical_dim + 1))], dtype="int32")

    return dense, cat, rowptr


class PredictHugeCTR(PipelineableInferenceOperator):
    def __init__(self, model, input_schema: Schema, *, backend="python", **hugectr_params):
        """Instantiate a HugeCTR inference operator.

        Parameters
        ----------
        model : HugeCTR Model Instance
            A HugeCTR model class.
        input_schema : merlin.schema.Schema
            The schema representing the input columns expected by the model.
        backend : str
            The Triton backend to use to when running this operator.
        **hugectr_params
            The parameters to pass to the HugeCTR operator.
        """
        if model is not None:
            self.hugectr_op = HugeCTR(model, **hugectr_params)

        self.backend = backend
        self.input_schema = input_schema
        self._hugectr_model_name = None

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Return the output schema representing the columns this operator returns."""
        return self.hugectr_op.compute_output_schema(
            input_schema, col_selector, prev_output_schema=prev_output_schema
        )

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """Return the input schema representing the input columns this operator expects to use."""
        return self.input_schema

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        """Export the class and related files to the path specified."""
        hugectr_model_config = self.hugectr_op.export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
        )
        params = params or {}
        params = {
            **params,
            "hugectr_model_name": hugectr_model_config.name,
            "slot_sizes": hugectr_model_config.parameters["slot_sizes"].string_value,
        }
        return super().export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
            backend=self.backend,
        )

    @classmethod
    def from_config(cls, config: dict) -> "PredictHugeCTR":
        """Instantiate the class from a dictionary representation.

        Expected structure:
        {
            "input_dict": str  # JSON dict with input names and schemas
            "params": str  # JSON dict with params saved at export
        }

        """

        column_schemas = [
            ColumnSchema(name, **schema_properties)
            for name, schema_properties in json.loads(config["input_dict"]).items()
        ]

        input_schema = Schema(column_schemas)

        cls_instance = cls(None, input_schema)
        params = json.loads(config["params"])

        cls_instance.slot_sizes = json.loads(params["slot_sizes"])
        cls_instance.set_hugectr_model_name(params["hugectr_model_name"])
        return cls_instance

    @property
    def hugectr_model_name(self):
        return self._hugectr_model_name

    def set_hugectr_model_name(self, hugectr_model_name):
        self._hugectr_model_name = hugectr_model_name

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """Transform the dataframe by applying this FIL operator to the set of input columns.

        Parameters
        -----------
        df: InferenceDataFrame
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        InferenceDataFrame
            Returns a transformed dataframe for this operator"""
        slot_sizes = [slot for slots in self.slot_sizes for slot in slots]
        categorical_columns = self.input_schema.select_by_tag(Tags.CATEGORICAL).column_names
        dict_to_pd = {k: v.ravel() for k, v in df}

        df = make_df(dict_to_pd)
        dense, cats, rowptr = _convert(df, slot_sizes, categorical_columns, labels=["label"])

        inputs = [
            pb_utils.Tensor("DES", dense),
            pb_utils.Tensor("CATCOLUMN", cats),
            pb_utils.Tensor("ROWINDEX", rowptr),
        ]

        inference_request = pb_utils.InferenceRequest(
            model_name=self.hugectr_model_name,
            requested_output_names=["OUTPUT0"],
            inputs=inputs,
        )
        inference_response = inference_request.exec()
        output0 = pb_utils.get_output_tensor_by_name(inference_response, "OUTPUT0")

        return InferenceDataFrame({"OUTPUT0": output0})


class HugeCTR(InferenceOperator):
    """
    Creates an operator meant to house a HugeCTR model.
    Allows the model to run as part of a merlin graph operations for inference.
    """

    def __init__(
        self,
        model,
        max_batch_size=64,
        device_list=None,
        hit_rate_threshold=None,
        gpucache=None,
        freeze_sparse=None,
        gpucacheper=None,
        max_nnz=2,
        embeddingkey_long_type=None,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.device_list = device_list or []
        embeddingkey_long_type = embeddingkey_long_type or "true"
        gpucache = gpucache or "true"
        gpucacheper = gpucacheper or 0.5

        self.hugectr_params = dict(
            hit_rate_threshold=hit_rate_threshold,
            gpucache=gpucache,
            freeze_sparse=freeze_sparse,
            gpucacheper=gpucacheper,
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
        model_name = node_name
        hugectr_model_path = pathlib.Path(node_export_path) / str(version)
        hugectr_model_path.mkdir(exist_ok=True)

        network_file = os.path.join(hugectr_model_path, f"{model_name}.json")

        self.model.graph_to_json(graph_config_file=network_file)
        self.model.save_params_to_files(str(hugectr_model_path) + "/")
        model_json = json.loads(open(network_file, "r").read())
        dense_pattern = "*_dense_*.model"
        dense_path = [
            os.path.join(hugectr_model_path, path.name)
            for path in hugectr_model_path.glob(dense_pattern)
            if "opt" not in path.name
        ][0]
        sparse_pattern = "*_sparse_*.model"
        sparse_paths = [
            os.path.join(hugectr_model_path, path.name)
            for path in hugectr_model_path.glob(sparse_pattern)
            if "opt" not in path.name
        ]

        config_dict = dict()
        config_dict["supportlonglong"] = True

        data_layer = model_json["layers"][0]
        sparse_layers = [
            layer
            for layer in model_json["layers"]
            if layer["type"] == "DistributedSlotSparseEmbeddingHash"
        ]
        full_slots = [x["sparse_embedding_hparam"]["slot_size_array"] for x in sparse_layers]
        num_cat_columns = sum(x["slot_num"] for x in data_layer["sparse"])
        vec_size = [x["sparse_embedding_hparam"]["embedding_vec_size"] for x in sparse_layers]

        model = dict()
        model["model"] = model_name
        model["slot_num"] = num_cat_columns
        model["sparse_files"] = sparse_paths
        model["dense_file"] = dense_path
        model["maxnum_des_feature_per_sample"] = data_layer["dense"]["dense_dim"]
        model["network_file"] = network_file
        model["num_of_worker_buffer_in_pool"] = 4
        model["num_of_refresher_buffer_in_pool"] = 1
        model["deployed_device_list"] = self.device_list
        model["max_batch_size"] = self.max_batch_size
        model["default_value_for_each_table"] = [0.0] * len(sparse_layers)
        model["hit_rate_threshold"] = 0.9
        model["gpucacheper"] = self.hugectr_params["gpucacheper"]
        model["gpucache"] = True
        model["cache_refresh_percentage_per_iteration"] = 0.2
        model["maxnum_catfeature_query_per_table_per_sample"] = [
            len(x["sparse_embedding_hparam"]["slot_size_array"]) for x in sparse_layers
        ]
        model["embedding_vecsize_per_table"] = vec_size
        model["embedding_table_names"] = [x["top"] for x in sparse_layers]
        config_dict["models"] = [model]

        parameter_server_config_path = str(node_export_path.parent / "ps.json")
        with open(parameter_server_config_path, "w") as f:
            f.write(json.dumps(config_dict))

        self.hugectr_params["config"] = network_file

        # These are no longer required from hugectr_backend release 3.7
        self.hugectr_params["cat_feature_num"] = num_cat_columns
        self.hugectr_params["des_feature_num"] = data_layer["dense"]["dense_dim"]
        self.hugectr_params["embedding_vector_size"] = vec_size[0]
        self.hugectr_params["slots"] = num_cat_columns
        self.hugectr_params["label_dim"] = data_layer["label"]["label_dim"]
        self.hugectr_params["slot_sizes"] = full_slots
        config = _hugectr_config(node_name, self.hugectr_params, max_batch_size=self.max_batch_size)

        with open(os.path.join(node_export_path, "config.pbtxt"), "w") as o:
            text_format.PrintMessage(config, o)

        return config


def _hugectr_config(name, parameters, max_batch_size=None):
    """Create a config for a HugeCTR model.

    Parameters
    ----------
    name : string
        The name of the hugectr model.
    parameters : dictionary
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

    for parameter_key, parameter_value in parameters.items():
        if parameter_value is None:
            continue

        if isinstance(parameter_value, list):
            config.parameters[parameter_key].string_value = json.dumps(parameter_value)
        elif isinstance(parameter_value, bool):
            config.parameters[parameter_key].string_value = str(parameter_value).lower()
        config.parameters[parameter_key].string_value = str(parameter_value)

    return config
