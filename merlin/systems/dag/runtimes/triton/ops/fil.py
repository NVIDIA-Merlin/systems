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
import pathlib

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.runtimes.triton.ops.operator import TritonOperator  # noqa
from merlin.systems.triton.conversions import (
    tensor_table_to_triton_request,
    triton_response_to_tensor_table,
)
from merlin.table import TensorTable


class PredictForestTriton(TritonOperator):
    """Operator for running inference on Forest models.

    This works for gradient-boosted decision trees (GBDTs) and Random forests (RF).
    While RF and GBDT algorithms differ in the way they train the models,
    they both produce a decision forest as their output.

    Uses the Forest Inference Library (FIL) backend for inference.
    """

    # def __init__(self, model, input_schema, *, backend="python", **fil_params):
    def __init__(self, op, input_schema=None):

        """Instantiate a FIL inference operator.

        Parameters
        ----------
        model : Forest Model Instance
            A forest model class. Supports XGBoost, LightGBM, and Scikit-Learn.
        input_schema : merlin.schema.Schema
            The schema representing the input columns expected by the model.
        backend : str
            The Triton backend to use to when running this operator.
        **fil_params
            The parameters to pass to the FIL operator.
        """
        super().__init__(op)
        if op is not None:
            self.fil_op = FILTriton(op.fil_op)
            self.backend = op.backend
            self.input_schema = op.input_schema
        if input_schema is not None:
            self.input_schema = input_schema
        self._fil_model_name = None

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Return the output schema representing the columns this operator returns."""
        return self.fil_op.compute_output_schema(
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

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
    ):
        """Export the class and related files to the path specified."""
        fil_model_config = self.fil_op.export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
        )
        self.set_fil_model_name(fil_model_config.name)
        params = params or {}
        params = {**params, "fil_model_name": self.fil_model_name}
        return super().export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
        )

    @property
    def fil_model_name(self):
        return self._fil_model_name

    def set_fil_model_name(self, fil_model_name):
        self._fil_model_name = fil_model_name

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this FIL operator to the set of input columns.

        Parameters
        -----------
        df: TensorTable
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        TensorTable
            Returns a transformed dataframe for this operator"""

        input0 = (
            np.array([column.values.ravel() for column in transformable.values()])
            .astype(np.float32)
            .T
        )

        inputs = TensorTable({"input__0": input0})
        input_schema = Schema(["input__0"])
        output_schema = Schema(["output__0"])

        inference_request = tensor_table_to_triton_request(
            self.fil_model_name, inputs, input_schema, output_schema
        )
        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise RuntimeError(str(inference_response.error().message()))

        return triton_response_to_tensor_table(inference_response, type(inputs), output_schema)


class FILTriton(TritonOperator):
    """Operator for Forest Inference Library (FIL) models.

    Packages up XGBoost models to run on Triton inference server using the fil backend.
    """

    def __init__(self, op):
        """Instantiate a FIL inference operator.

        Parameters
        ----------
        model : Forest model
            A forest model class. Supports XGBoost, LightGBM, and Scikit-Learn.
        max_batch_size : int
           The maximum number of samples to process in a batch. In general, FIL's
           efficient handling of even large forest models means that this value can be
           quite high, but this may need to be reduced for your particular hardware
           configuration if you find that you are exhausting system resources (such as GPU
           or system RAM).
        predict_proba : bool
            If using a classification model. Specifies whether the desired output is a
            score for each class or merely the predicted class ID. Changes the output size
            to NUMBER_OF_CLASSES.
        output_class : bool
            Is the model a classification model? If set to True will output class ID,
            unless predict_proba is also to True.
        threshold : float
            If using a classification model. The threshold score used for class
            prediction. Defaults to 0.5.
        algo : str
            One of "ALGO_AUTO", "NAIVE", "TREE_REORG" or "BATCH_TREE_REORG" indicating
            which FIL inference algorithm to use. More details are available in the cuML
            documentation. If you are uncertain of what algorithm to use, we recommend
            selecting "ALGO_AUTO", since it is a safe choice for all models.
        storage_type : str
            One of "AUTO", "DENSE", "SPARSE", and "SPARSE8", indicating the storage format
            that should be used to represent the imported model. "AUTO" indicates that the
            storage format should be automatically chosen. "SPARSE8" is currently
            experimental.
        threads_per_tree : int
            Determines number of threads used to use for inference on a single
            tree. Increasing this above 1 can improve memory bandwidth near the tree root
            but use more shared memory. In general, network latency will significantly
            overshadow any speedup from tweaking this setting, but it is provided for
            cases where maximizing throughput is essential. for a more thorough
            explanation of this parameter and how it may be used.
        blocks_per_sm : int
             If set to any nonzero value (generally between 2 and 7), this provides a
             limit to improve the cache hit rate for large forest models. In general,
             network latency will significantly overshadow any speedup from tweaking this
             setting, but it is provided for cases where maximizing throughput is
             essential. Please see the cuML documentation for a more thorough explanation
             of this parameter and how it may be used.
        transfer_threshold : int
             If the number of samples in a batch exceeds this value and the model is
             deployed on the GPU, then GPU inference will be used. Otherwise, CPU
             inference will be used for batches received in host memory with a number of
             samples less than or equal to this threshold. For most models and systems,
             the default transfer threshold of 0 (meaning that data is always transferred
             to the GPU for processing) will provide optimal latency and throughput, but
             for low-latency deployments with the use_experimental_optimizations flag set
             to true, higher values may be desirable.
        instance_group : str
             One of "AUTO", "GPU", "CPU". Default value is "AUTO". Specifies whether
             inference will take place on the GPU or CPU.
        """
        self.max_batch_size = op.max_batch_size
        self.parameters = dict(**op.parameters)
        self.fil_model_class = op.fil_model_class
        super().__init__(op)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "fil_model"}

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """Returns input schema for FIL op"""
        return self.op.compute_input_schema(root_schema, parents_schema, deps_schema, selector)

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Returns output schema for FIL op"""
        return self.op.compute_output_schema(input_schema, col_selector, prev_output_schema)

    def export(
        self,
        path,
        input_schema,
        output_schema,
        params: dict = None,
        node_id=None,
        version=1,
    ):
        """Export the model to the supplied path. Returns the config"""
        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name
        node_export_path = pathlib.Path(path) / node_name
        version_path = node_export_path / str(version)
        version_path.mkdir(parents=True, exist_ok=True)

        self.fil_model_class.save(version_path)

        config = fil_config(
            node_name,
            self.fil_model_class.model_type,
            self.fil_model_class.num_features,
            self.fil_model_class.num_classes,
            max_batch_size=self.max_batch_size,
            **self.parameters,
        )

        with open(node_export_path / "config.pbtxt", "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        return config


def fil_config(
    name,
    model_type,
    num_features,
    num_classes,
    *,
    max_batch_size=8192,
    predict_proba=False,
    output_class=False,
    threshold=0.5,
    algo="ALGO_AUTO",
    storage_type="AUTO",
    blocks_per_sm=0,
    threads_per_tree=1,
    transfer_threshold=0,
    instance_group="AUTO",
) -> model_config.ModelConfig:
    """Construct and return a FIL ModelConfig protobuf object.

    Parameters
    ----------
    name : str
        The name of the model
    model_type : str
        The type of model. One of {xgboost, xgboost_json, lightgbm, treelite_checkpoint}
    num_features : int
        The number of input features to the model.
    num_classes : int
        If the model is a classifier. The number of classes.
    max_batch_size : int
       The maximum number of samples to process in a batch. In general, FIL's
       efficient handling of even large forest models means that this value can be
       quite high, but this may need to be reduced for your particular hardware
       configuration if you find that you are exhausting system resources (such as GPU
       or system RAM).
    predict_proba : bool
        If using a classification model. Specifies whether the desired output is a
        score for each class or merely the predicted class ID. Changes the output size
        to NUMBER_OF_CLASSES.
    output_class : bool
        Is the model a classification model? If set to True will output class ID,
        unless predict_proba is also to True.
    threshold : float
        If using a classification model. The threshold score used for class
        prediction. Defaults to 0.5.
    algo : str
        One of "ALGO_AUTO", "NAIVE", "TREE_REORG" or "BATCH_TREE_REORG" indicating
        which FIL inference algorithm to use. More details are available in the cuML
        documentation. If you are uncertain of what algorithm to use, we recommend
        selecting "ALGO_AUTO", since it is a safe choice for all models.
    storage_type : str
        One of "AUTO", "DENSE", "SPARSE", and "SPARSE8", indicating the storage format
        that should be used to represent the imported model. "AUTO" indicates that the
        storage format should be automatically chosen. "SPARSE8" is currently
        experimental.
    threads_per_tree : int
        Determines number of threads used to use for inference on a single
        tree. Increasing this above 1 can improve memory bandwidth near the tree root
        but use more shared memory. In general, network latency will significantly
        overshadow any speedup from tweaking this setting, but it is provided for
        cases where maximizing throughput is essential. for a more thorough
        explanation of this parameter and how it may be used.
    blocks_per_sm : int
         If set to any nonzero value (generally between 2 and 7), this provides a
         limit to improve the cache hit rate for large forest models. In general,
         network latency will significantly overshadow any speedup from tweaking this
         setting, but it is provided for cases where maximizing throughput is
         essential. Please see the cuML documentation for a more thorough explanation
         of this parameter and how it may be used.
    transfer_threshold : int
         If the number of samples in a batch exceeds this value and the model is
         deployed on the GPU, then GPU inference will be used. Otherwise, CPU
         inference will be used for batches received in host memory with a number of
         samples less than or equal to this threshold. For most models and systems,
         the default transfer threshold of 0 (meaning that data is always transferred
         to the GPU for processing) will provide optimal latency and throughput, but
         for low-latency deployments with the use_experimental_optimizations flag set
         to true, higher values may be desirable.
    instance_group : str
         One of "AUTO", "GPU", "CPU". Default value is "AUTO". Specifies whether
         inference will take place on the GPU or CPU.

    Returns
        model_config.ModelConfig
    """
    input_dim = num_features
    output_dim = 1

    # if we have multiclass then switch to output_class
    if num_classes > 2:
        output_class = True

    if output_class and predict_proba:
        output_dim = num_classes

        # where classifier has not specified the number of classes
        # and we're requesting to output class probabilities
        # assume this is a binary classifier
        output_dim = max(output_dim, 2)

    parameters = {
        "model_type": model_type,
        "predict_proba": "true" if predict_proba else "false",
        "output_class": "true" if output_class else "false",
        "threshold": f"{threshold:.4f}",
        "storage_type": storage_type,
        "algo": algo,
        "use_experimental_optimizations": "false",
        "blocks_per_sm": f"{blocks_per_sm:d}",
        "threads_per_tree": f"{threads_per_tree:d}",
        "transfer_threshold": f"{transfer_threshold:d}",
    }

    supported_instance_groups = {"auto", "cpu", "gpu"}
    instance_group = instance_group.lower() if isinstance(instance_group, str) else instance_group
    if instance_group == "auto":
        instance_group_kind = model_config.ModelInstanceGroup.Kind.KIND_AUTO
    elif instance_group == "cpu":
        instance_group_kind = model_config.ModelInstanceGroup.Kind.KIND_CPU
    elif instance_group == "gpu":
        instance_group_kind = model_config.ModelInstanceGroup.Kind.KIND_GPU
    else:
        raise ValueError(f"instance_group must be one of {supported_instance_groups}")

    config = model_config.ModelConfig(
        name=name,
        backend="fil",
        max_batch_size=max_batch_size,
        input=[
            model_config.ModelInput(
                name="input__0",
                data_type=model_config.TYPE_FP32,
                dims=[input_dim],
            )
        ],
        output=[
            model_config.ModelOutput(
                name="output__0", data_type=model_config.TYPE_FP32, dims=[output_dim]
            )
        ],
        instance_group=[model_config.ModelInstanceGroup(kind=instance_group_kind)],
    )

    for parameter_key, parameter_value in parameters.items():
        config.parameters[parameter_key].string_value = parameter_value

    return config
