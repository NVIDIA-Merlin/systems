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
import pathlib
import pickle
from abc import ABC, abstractmethod

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.dag import ColumnSelector  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.ops.compat import (
    cuml_ensemble,
    lightgbm,
    pb_utils,
    sklearn_ensemble,
    xgboost,
)
from merlin.systems.dag.ops.operator import (
    InferenceDataFrame,
    InferenceOperator,
    PipelineableInferenceOperator,
)


class PredictForest(PipelineableInferenceOperator):
    """Operator for running inference on Forest models.

    This works for gradient-boosted decision trees (GBDTs) and Random forests (RF).
    While RF and GBDT algorithms differ in the way they train the models,
    they both produce a decision forest as their output.

    Uses the Forest Inference Library (FIL) backend for inference.
    """

    def __init__(self, model, input_schema, *, backend="python", **fil_params):
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
        if model is not None:
            self.fil_op = FIL(model, **fil_params)
        self.backend = backend
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

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        """Export the class and related files to the path specified."""
        fil_model_config = self.fil_op.export(
            path,
            input_schema,
            output_schema,
            params=params,
            node_id=node_id,
            version=version,
        )
        params = params or {}
        params = {**params, "fil_model_name": fil_model_config.name}
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
    def from_config(cls, config: dict) -> "PredictForest":
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
        cls_instance.set_fil_model_name(params["fil_model_name"])
        return cls_instance

    @property
    def fil_model_name(self):
        return self._fil_model_name

    def set_fil_model_name(self, fil_model_name):
        self._fil_model_name = fil_model_name

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
        input0 = np.array([x.ravel() for x in df.tensors.values()]).astype(np.float32).T
        inference_request = pb_utils.InferenceRequest(
            model_name=self.fil_model_name,
            requested_output_names=["output__0"],
            inputs=[pb_utils.Tensor("input__0", input0)],
        )
        inference_response = inference_request.exec()
        output0 = pb_utils.get_output_tensor_by_name(inference_response, "output__0")
        return InferenceDataFrame({"output__0": output0})


class FIL(InferenceOperator):
    """Operator for Forest Inference Library (FIL) models.

    Packages up XGBoost models to run on Triton inference server using the fil backend.
    """

    def __init__(
        self,
        model,
        *,
        max_batch_size=8192,
        predict_proba=False,
        output_class=False,
        threshold=0.5,
        algo="ALGO_AUTO",
        storage_type="AUTO",
        threads_per_tree=1,
        blocks_per_sm=0,
        transfer_threshold=0,
        instance_group="AUTO",
    ):
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
        self.max_batch_size = max_batch_size
        self.parameters = dict(
            predict_proba=predict_proba,
            output_class=output_class,
            algo=algo,
            transfer_threshold=transfer_threshold,
            blocks_per_sm=blocks_per_sm,
            storage_type=storage_type,
            threshold=threshold,
            instance_group=instance_group,
        )
        self.fil_model = get_fil_model(model)
        super().__init__()

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """Returns input schema for FIL op"""
        return Schema([ColumnSchema("input__0", dtype=np.float32)])

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Returns output schema for FIL op"""
        return Schema([ColumnSchema("output__0", dtype=np.float32)])

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

        self.fil_model.save(version_path)

        config = fil_config(
            node_name,
            self.fil_model.model_type,
            self.fil_model.num_features,
            self.fil_model.num_classes,
            max_batch_size=self.max_batch_size,
            **self.parameters,
        )

        with open(node_export_path / "config.pbtxt", "w") as o:
            text_format.PrintMessage(config, o)

        return config


class FILModel(ABC):
    """Interface for a FIL Model. Methods implement required information to construct a FIL
    model configuration file"""

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def save(self, version_path):
        """
        Save model to version_path
        """

    @property
    @abstractmethod
    def num_classes(self):
        """
        The number of classes
        """

    @property
    @abstractmethod
    def num_features(self):
        """
        The number of features
        """

    @property
    @abstractmethod
    def num_targets(self):
        """
        The number of targets
        """

    @property
    @abstractmethod
    def model_type(self):
        """The type of model"""


def get_fil_model(model) -> FILModel:
    """Return FILModel class corresponding to the model passed in.

    Parameters
    ----------
    model
        A model class.

        Supports:
            - XGBoost {Booster, XGBClassifier, XGBRegressor}
            - LightGBM {Booster}
            - Scikit-Learn {RandomForestClassifier, RandomForestRegressor}
    """
    if xgboost and isinstance(model, xgboost.Booster):
        fil_model = XGBoost(model)
    elif xgboost and isinstance(model, xgboost.XGBModel):
        fil_model = XGBoost(model.get_booster())
    elif lightgbm and isinstance(model, lightgbm.Booster):
        fil_model = LightGBM(model)
    elif lightgbm and isinstance(model, lightgbm.LGBMModel):
        fil_model = LightGBM(model.booster_)
    elif cuml_ensemble and isinstance(
        model,
        (
            cuml_ensemble.RandomForestClassifier,
            cuml_ensemble.RandomForestRegressor,
        ),
    ):
        fil_model = CUMLRandomForest(model)
    elif sklearn_ensemble and isinstance(
        model,
        (
            sklearn_ensemble.RandomForestClassifier,
            sklearn_ensemble.RandomForestRegressor,
        ),
    ):
        fil_model = SKLearnRandomForest(model)
    else:
        supported_model_types = {
            "xgboost.Booster",
            "xgboost.XGBModel",
            "lightgbm.Booster",
            "lightgbm.LGBMModel",
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.ensemble.RandomForestRegressor",
            "cuml.ensemble.RandomForestClassifier",
            "cuml.ensemble.RandomForestRegressor",
        }
        raise ValueError(
            f"Model type not supported. {type(model)} " f"Must be one of: {supported_model_types}"
        )
    return fil_model


class XGBoost(FILModel):
    """XGBoost Wrapper for FIL."""

    model_type = "xgboost_json"
    model_filename = "xgboost.json"

    def __init__(self, model):
        self.model = model
        learner = json.loads(model.save_config())["learner"]
        self._learner = learner

        objective = learner["objective"]["name"]
        if objective == "binary:hinge":
            raise ValueError(
                "Objective binary:hinge is not supported."
                "Only sigmoid and identity values of pred_transform are supported"
                " for binary classification."
            )

        learner_model_param = learner["learner_model_param"]
        num_targets = int(learner_model_param["num_target"])

        if num_targets > 1:
            raise ValueError("Only single target objectives are supported.")

        super().__init__(model)

    def save(self, version_path) -> None:
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        self.model.save_model(model_path)

    def _get_learner(self):
        return self._learner

    @property
    def num_classes(self):
        learner = self._get_learner()
        learner_model_param = learner["learner_model_param"]
        num_classes = int(learner_model_param["num_class"])
        return num_classes

    @property
    def num_features(self):
        return self.model.num_features()

    @property
    def num_targets(self):
        learner = self._get_learner()
        learner_model_param = learner["learner_model_param"]
        num_targets = int(learner_model_param["num_target"])
        return num_targets


class LightGBM(FILModel):
    """LightGBM Wrapper for FIL."""

    model_type = "lightgbm"
    model_filename = "model.txt"

    def save(self, version_path):
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        self.model.save_model(model_path)

    @property
    def num_features(self):
        return self.model.num_feature()

    @property
    def num_classes(self):
        return self.model.dump_model()["num_class"]

    @property
    def num_targets(self):
        # only supports one target
        return 1


class SKLearnRandomForest(FILModel):
    """Scikit-Learn RandomForest Wrapper for FIL."""

    model_type = "treelite_checkpoint"
    model_filename = "model.pkl"

    def save(self, version_path):
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        with open(model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)

    @property
    def num_features(self):
        return self.model.n_features_in_

    @property
    def num_classes(self):
        try:
            return self.model.n_classes_
        except AttributeError:
            # n_classes_ is not defined for RandomForestRegressor
            return 0

    @property
    def num_targets(self):
        return self.model.n_outputs_


class CUMLRandomForest(FILModel):

    model_type = "treelite_checkpoint"
    model_filename = "model.pkl"

    def save(self, version_path):
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        with open(model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)

    @property
    def num_features(self):
        return self.model.n_features_in_

    @property
    def num_classes(self):
        try:
            return self.model.num_classes
        except AttributeError:
            # num_classes is not defined for RandomForestRegressor
            return 0

    @property
    def num_targets(self):
        # Only supports one target
        return 1


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
