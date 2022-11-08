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
import importlib
import json
import pathlib

import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import PipelineableInferenceOperator

try:
    import implicit
    from packaging.version import Version

    if Version(implicit.__version__) < Version("0.6.0"):
        raise RuntimeError(
            "Implicit version 0.6.0 or higher required. (for model save/load methods)."
        )
except ImportError:
    implicit = None


class PredictImplicit(PipelineableInferenceOperator):
    """Operator for running inference on Implicit models.."""

    def __init__(self, model, num_to_recommend: int = 10, **kwargs):
        """Instantiate an Implicit prediction operator.

        Parameters
        ----------
        model : An Implicit Model instance
        num_to_recommend : int
           the number of items to return
        """
        self.model = model
        self.model_module_name: str = self.model.__module__
        self.model_class_name: str = self.model.__class__.__name__
        self.num_to_recommend = num_to_recommend
        super().__init__(**kwargs)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "model"}

    def load_artifacts(self, artifact_path: str):
        model_file = pathlib.Path(artifact_path) / "model.npz"

        model_module_name = self.model_module_name
        model_class_name = self.model_class_name
        model_module = importlib.import_module(model_module_name)
        model_cls = getattr(model_module, model_class_name)

        self.model = model_cls.load(str(model_file))

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """Return the input schema representing the input columns this operator expects to use."""
        return Schema([ColumnSchema("user_id", dtype="int64")])

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Return the output schema representing the columns this operator returns."""
        return Schema([ColumnSchema("ids", dtype="int64"), ColumnSchema("scores", dtype="float64")])

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
        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name

        if backend == "ensemble":
            artifact_path = pathlib.Path(path) / node_name / str(version)
        else:
            artifact_path = pathlib.Path(path) / "executor_model" / str(version) / "ensemble"

        artifact_path.mkdir(parents=True, exist_ok=True)
        model_path = artifact_path / "model.npz"
        self.model.save(str(model_path))

        if backend == "ensemble":
            params = params or {}
            params["model_module_name"] = self.model.__module__
            params["model_class_name"] = self.model.__class__.__name__
            params["num_to_recommend"] = self.num_to_recommend
            return super().export(
                path,
                input_schema,
                output_schema,
                params=params,
                node_id=node_id,
                version=version,
            )
        else:
            return ({}, [])

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "PredictImplicit":
        """Instantiate the class from a dictionary representation.

        Expected config structure:
        {
            "input_dict": str  # JSON dict with input names and schemas
            "params": str  # JSON dict with params saved at export
        }

        """
        params = json.loads(config["params"])

        model_repository = kwargs["model_repository"]
        model_name = kwargs["model_name"]
        model_version = kwargs["model_version"]

        # load implicit model
        model_module_name = params["model_module_name"]
        model_class_name = params["model_class_name"]
        model_module = importlib.import_module(model_module_name)
        model_cls = getattr(model_module, model_class_name)
        model_file = pathlib.Path(model_repository) / model_name / str(model_version) / "model.npz"
        model = model_cls.load(str(model_file))

        num_to_recommend = params["num_to_recommend"]

        return cls(model, num_to_recommend=num_to_recommend)

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this operator to the set of input columns.

        Parameters
        -----------
        df: DictArray
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DictArray
            Returns a transformed dataframe for this operator"""
        user_id = transformable["user_id"].values.ravel()
        user_items = None
        ids, scores = self.model.recommend(
            user_id, user_items, N=self.num_to_recommend, filter_already_liked_items=False
        )
        return type(transformable)(
            {"ids": ids.astype(np.int64), "scores": scores.astype(np.float64)}
        )
