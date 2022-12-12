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
from abc import ABC, abstractmethod

import numpy as np

from merlin.core.dispatch import HAS_GPU
from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.dictarray import DictArray
from merlin.systems.dag.ops.compat import (
    cuml_ensemble,
    cuml_fil,
    lightgbm,
    sklearn_ensemble,
    treelite_model,
    treelite_sklearn,
    xgboost,
)
from merlin.systems.dag.ops.operator import InferenceOperator, PipelineableInferenceOperator


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
        super().__init__()

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

    @property
    def exportable_backends(self):
        return ["ensemble", "executor"]

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
        backend: str = "ensemble",
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

        return fil_model_config

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
        df: DictArray
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DictArray
            Returns a transformed dataframe for this operator"""

        input0 = (
            np.array([column.values.ravel() for column in transformable.values()])
            .astype(np.float32)
            .T
        )
        predictions = self.fil_op.predict(input0).astype(np.float32)

        outputs = {"output__0": predictions}

        return type(transformable)(outputs)

    def load_artifacts(self, artifact_path):
        # need variable that tells me what type of model this is.
        self.fil_op.load_model(artifact_path)


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
        self.fil_model_class = get_fil_model_class(model)
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

        self.fil_model_class.save(version_path)

        return version_path

    def load_model(self, version_path):
        version_path = pathlib.Path(version_path)
        self.fil_model_class.model = self.fil_model_class.load(version_path)

    def predict(self, inputs):
        return self.fil_model_class.predict(inputs)


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

    def predict(self, inputs):
        return self.model.predict(inputs)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "model"}


def get_fil_model_class(model) -> FILModel:
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
    elif xgboost and hasattr(model, "booster"):
        # support the merlin.models.xgb.XGBoost wrapper too
        fil_model = XGBoost(model.booster)
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
            "merlin.models.xgb.XGBoost",
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
        if model:
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

    def predict(self, inputs):
        if isinstance(inputs, DictArray):
            inputs = inputs.to_df()
        inputs = xgboost.DMatrix(inputs)
        return self.model.predict(inputs)

    def save(self, version_path) -> None:
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        self.model.save_model(model_path)

    def load(self, version_path) -> "XGBoost":
        model_path = pathlib.Path(version_path) / self.model_filename
        if HAS_GPU:
            self.model = cuml_fil.load(model_path, output_class=True)
        else:
            self.model = treelite_model.load(model_path, self.model_type)

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

    def load(self, version_path) -> "LightGBM":
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        if HAS_GPU:
            self.model = cuml_fil.load(model_path, output_class=True)
        else:
            self.model = treelite_model.load(model_path, self.model_type)

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
    model_filename = "checkpoint.tl"

    def save(self, version_path):
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        if treelite_sklearn is None:
            raise RuntimeError(
                "Both 'treelite' and 'treelite_runtime' "
                "are required to save an sklearn random forest model."
            )
        treelite_model = treelite_sklearn.import_model(self.model)
        treelite_model.serialize(str(model_path))

    def load(self, version_path) -> "SKLearnRandomForest":
        model_path = pathlib.Path(version_path) / self.model_filename
        if treelite_sklearn is None:
            raise RuntimeError(
                "Both 'treelite' and 'treelite_runtime' "
                "are required to save an sklearn random forest model."
            )
        self.model = treelite_model.deserialize(str(model_path))

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
    model_filename = "checkpoint.tl"

    def save(self, version_path):
        """Save model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        self.model.convert_to_treelite_model().to_treelite_checkpoint(str(model_path))

    def load(self, version_path) -> "CUMLRandomForest":
        """Load model to version_path."""
        model_path = pathlib.Path(version_path) / self.model_filename
        if HAS_GPU:
            self.model = cuml_fil.load(model_path, output_class=True)
        else:
            self.model = treelite_model.deserialize(model_path)

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
