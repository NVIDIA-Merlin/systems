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
import pathlib

import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceOperator

try:
    import implicit
    from packaging.version import Version

    if Version(implicit.__version__) < Version("0.6.0"):
        raise RuntimeError(
            "Implicit version 0.6.0 or higher required. (for model save/load methods)."
        )
except ImportError:
    implicit = None


class PredictImplicit(InferenceOperator):
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

    def save_artifacts(self, artifact_path: str):
        model_path = pathlib.Path(artifact_path) / "model.npz"
        self.model.save(str(model_path))

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

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this operator to the set of input columns.

        Parameters
        -----------
        df: TensorTable
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        TensorTable
            Returns a transformed dataframe for this operator"""
        user_id = transformable["user_id"].values.ravel()
        user_items = None
        ids, scores = self.model.recommend(
            user_id, user_items, N=self.num_to_recommend, filter_already_liked_items=False
        )
        return type(transformable)(
            {"ids": ids.astype(np.int64), "scores": scores.astype(np.float64)}
        )
