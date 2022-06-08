import json
import pathlib
import pickle
from abc import ABC, abstractmethod

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.dag import ColumnSelector  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.ops.compat import cuml_ensemble, lightgbm, sklearn_ensemble, xgboost
from merlin.systems.dag.ops.operator import InferenceOperator


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
        instance_group=[
            model_config.ModelInstanceGroup(kind=model_config.ModelInstanceGroup.Kind.KIND_AUTO)
        ],
    )

    for parameter_key, parameter_value in parameters.items():
        config.parameters[parameter_key].string_value = parameter_value

    return config
