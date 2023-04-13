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
import os
import pathlib
import tempfile

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from merlin.core.compat.tensorflow import tensorflow as tf  # noqa
from merlin.core.protocols import Transformable  # noqa
from merlin.dag import ColumnSelector  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.ops.operator import InferenceOperator  # noqa
from merlin.table import TensorflowColumn, TensorTable  # noqa
from merlin.table.conversions import convert_col  # noqa


class PredictTensorflow(InferenceOperator):
    """TensorFlow Model Prediction Operator."""

    def __init__(self, model_or_path, custom_objects: dict = None):
        """
        Instantiate a PredictTensorflow inference operator.

        Parameters
        ----------
        model_or_path : Tensorflow model or string
            This can be a tensorflow model or a path to a tensorflow model.
        custom_objects : dict, optional
            Any custom objects that need to be loaded with the model, by default None.
        """
        super().__init__()

        if model_or_path is not None:
            custom_objects = custom_objects or {}

            if isinstance(model_or_path, (str, os.PathLike)):
                self.path = model_or_path
                self.model = tf.keras.models.load_model(self.path, custom_objects=custom_objects)
            else:
                self.path = None
                self.model = model_or_path

            self.input_schema, self.output_schema = _construct_schemas_from_model(self.model)

    def __getstate__(self) -> dict:
        """Return state of instance when pickled.

        Returns
        -------
        dict
            Returns object state excluding model attribute.
        """
        return {k: v for k, v in self.__dict__.items() if k != "model"}

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Run model inference. Returning predictions.

        Parameters
        ----------
        col_selector : ColumnSelector
            Unused ColumunSelector input
        transformable : Transformable
            Input features to model

        Returns
        -------
        Transformable
            Model Predictions
        """
        # TODO: Validate that the inputs match the schema
        # TODO: Should we coerce the dtypes to match the schema here?
        if not isinstance(transformable, TensorTable):
            transformable = TensorTable.from_df(transformable)

        col_type = TensorflowColumn

        tf_columns = {
            col_name: convert_col(column, col_type) for col_name, column in transformable.items()
        }
        model_inputs = TensorTable(tf_columns)

        outputs = self.model(model_inputs.to_dict())

        dict_outputs = {}
        for col in self.output_schema.column_names:
            dict_outputs[col] = (
                outputs[col].numpy() if isinstance(outputs, dict) else outputs.numpy()
            )

        # TODO: map output schema names to outputs produced by prediction
        return type(transformable)(dict_outputs)

    @classmethod
    def from_path(cls, path, **kwargs):
        return cls.__init__(path, **kwargs)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Use the input schema supplied during object creation.
        """
        return self.input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """
        Use the output schema supplied during object creation.
        """
        return self.output_schema


def _construct_schemas_from_model(model):
    signatures = getattr(model, "signatures", {}) or {}
    default_signature = signatures.get("serving_default")

    if not default_signature:
        # roundtrip saved model to disk to generate signature if it doesn't exist
        model = _ensure_input_spec_includes_names(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tf_model_path = pathlib.Path(tmp_dir) / "model.savedmodel"
            model.save(tf_model_path, include_optimizer=False)
            reloaded = tf.keras.models.load_model(tf_model_path)
            default_signature = reloaded.signatures["serving_default"]

    input_schema = _build_schema_from_signature(default_signature.structured_input_signature[1])
    output_schema = _build_schema_from_signature(default_signature.structured_outputs)

    return input_schema, output_schema


def _ensure_input_spec_includes_names(model):
    if isinstance(model._saved_model_inputs_spec, dict):
        for key, spec in model._saved_model_inputs_spec.items():
            if isinstance(spec, tuple):
                model._saved_model_inputs_spec[key] = (
                    tf.TensorSpec(shape=spec[0].shape, dtype=spec[0].dtype, name=key),
                    tf.TensorSpec(shape=spec[1].shape, dtype=spec[1].dtype, name=key),
                )
            else:
                model._saved_model_inputs_spec[key] = tf.TensorSpec(
                    shape=spec.shape, dtype=spec.dtype, name=key
                )

    return model


def _build_schema_from_signature(signature):
    schema = Schema()
    for col_name, col in signature.items():
        if "__offsets" in col_name or "__values" in col_name:
            col_name = col_name.replace("__offsets", "").replace("__values", "")
            col_values_sig = signature[f"{col_name}__values"]
            col_offsets_sig = signature[f"{col_name}__offsets"]
            col_dtype = col_values_sig.dtype.as_numpy_dtype
            col_dims = (col_offsets_sig.shape[0], None)
        else:
            col_dtype = col.dtype.as_numpy_dtype
            col_dims = col.shape
        col_schema = ColumnSchema(col_name, dtype=col_dtype, dims=col_dims)
        schema.column_schemas[col_name] = col_schema
    return schema
