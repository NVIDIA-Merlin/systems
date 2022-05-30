import pathlib

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.dag import ColumnSelector  # noqa
from merlin.schema import ColumnSchema, Schema  # noqa
from merlin.systems.dag.ops.operator import InferenceOperator


class FIL(InferenceOperator):
    """Operator for Forest Inference Library (FIL) models.

    Packages up XGBoost models to run on Triton inference server using the fil backend.
    """

    def __init__(self, model):
        """Instantiate a FILPredict inference operator.

        Parameters
        ----------
        model : xgboost.Booster
            An XGBoost model.
        """
        self.model = model
        super().__init__()

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        return Schema([ColumnSchema("input__0", dtype=np.float32)])

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return Schema([ColumnSchema("output__0", dtype=np.float32)])

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Export the model to the supplied path."""
        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name
        node_export_path = pathlib.Path(path) / node_name
        version_path = node_export_path / str(version)
        version_path.mkdir(parents=True, exist_ok=True)
        model_path = version_path / "xgboost.json"

        self.model.save_model(model_path)

        config = model_config.ModelConfig(
            name=node_name,
            backend="fil",
            max_batch_size=8192,
            input=[
                model_config.ModelInput(
                    name="input__0",
                    data_type=model_config.TYPE_FP32,
                    dims=[self.model.num_features()],
                )
            ],
            output=[
                model_config.ModelOutput(
                    name="output__0", data_type=model_config.TYPE_FP32, dims=[1]
                )
            ],
            instance_group=[
                model_config.ModelInstanceGroup(kind=model_config.ModelInstanceGroup.Kind.KIND_AUTO)
            ],
        )

        default_parameters = {
            "model_type": "xgboost_json",
            "predict_proba": "true",
            "output_class": "true",
            "threshold": "0.5",
            "storage_type": "AUTO",
            "algo": "ALGO_AUTO",
            "use_experimental_optimizations": "false",
        }
        for parameter_key, parameter_value in default_parameters.items():
            config.parameters[parameter_key].string_value = parameter_value

        with open(node_export_path / "config.pbtxt", "w") as o:
            text_format.PrintMessage(config, o)

        return config
