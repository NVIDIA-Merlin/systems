import importlib.resources
import json
import os
import pathlib
from abc import abstractmethod
from shutil import copyfile

from merlin.systems.model_registry import ModelRegistry

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc.model_config_pb2 as model_config  # noqa
from google.protobuf import text_format  # noqa

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import BaseOperator  # noqa
from merlin.dag.selector import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.node import InferenceNode  # noqa
from merlin.systems.dag.ops import compute_dims  # noqa
from merlin.systems.triton.export import _convert_dtype  # noqa


class InferenceOperator(BaseOperator):
    """
    The basic building block of the Inference Node, the Inference Operator encapsulates
    a transform with the appropriate functionality to handle all Ensemble graph
    operations.
    """

    @property
    def export_name(self):
        """
        Provides a clear common english identifier for this operator.

        Returns
        -------
        String
            Name of the current class as spelled in module.
        """
        return self.__class__.__name__.lower()

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this operator to the set of input columns.

        This is defined here to force child classes to define a transform method, in order
        to avoid difficult to debug issues that surface in Triton with less-than-informative
        errors.

        Parameters
        -----------
        columns: list of str or list of list of str
            The columns to apply this operator to
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        raise NotImplementedError

    def load_artifacts(self, artifact_path):
        """
        Hook method that provides a way to load saved artifacts for the operator

        Parameters
        ----------
        artifact_path : str
            Path where artifacts for the operator are stored.
        """

    @property
    def exportable_backends(self):
        return ["ensemble"]

    @abstractmethod
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
        """
        Export the class object as a config and all related files to the user defined path.

        Parameters
        ----------
        path : str
            Artifact export path
        input_schema : Schema
            A schema with information about the inputs to this operator.
        output_schema : Schema
            A schema with information about the outputs of this operator.
        params : dict, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None.
        node_id : int, optional
            The placement of the node in the graph (starts at 1), by default None.
        version : int, optional
            The version of the model, by default 1.

        Returns
        -------
        Ensemble_config: dict
            The config for the entire ensemble.
        Node_configs: list
            A list of individual configs for each step (operator) in graph.
        """
        raise NotImplementedError

    def create_node(self, selector: ColumnSelector) -> InferenceNode:
        """_summary_

        Parameters
        ----------
        selector : ColumnSelector
            Selector to turn into an Inference Node.

        Returns
        -------
        InferenceNode
            New node for Ensemble graph.
        """
        return InferenceNode(selector)

    @classmethod
    def from_model_registry(cls, registry: ModelRegistry, **kwargs) -> "InferenceOperator":
        """
        Loads the InferenceOperator from the provided ModelRegistry.

        Parameters
        ----------
        registry : ModelRegistry
            A ModleRegistry object that will provide the path to the model.
        **kwargs
            Other kwargs to pass to your InferenceOperator's constructor.

        Returns
        -------
        InferenceOperator
            New node for Ensemble graph.
        """

        return cls.from_path(registry.get_artifact_uri(), **kwargs)

    @classmethod
    def from_path(cls, path, **kwargs) -> "InferenceOperator":
        """
        Loads the InferenceOperator from the path where it was exported after training.

        Parameters
        ----------
        path : str
            Path to the exported model.
        **kwargs
            Other kwargs to pass to your InferenceOperator's constructor.

        Returns
        -------
        InferenceOperator
            New node for Ensemble graph.
        """
        raise NotImplementedError(f"{cls.__name__} operators cannot be instantiated with a path.")

    # TODO: This operator should be moved once all triton specific op migrations completed
    @property
    def scalar_shape(self):
        return [1]


# TODO: This gets absorbed into TritonOperator after migration of all triton operators.
class PipelineableInferenceOperator(InferenceOperator):
    """
    This Inference operator type builds on the base infrence operator, by allowing
    chains of sequential pipelineable inference operators to be executed inside the
    same model. This remove tritonserver overhead between operators of this type.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, **kwargs):
        """
        Instantiate a class object given a config.

        Parameters
        ----------
        config : dict
        **kwargs
          contains the following:
            * model_repository: Model repository path
            * model_version: Model version
            * model_name: Model name

        Returns
        -------
            Class object instantiated with config values
        """

    @abstractmethod
    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        return transformable

    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
        backend: str = "python",
    ):
        """
        Export the class object as a config and all related files to the user-defined path.

        Parameters
        ----------
        path : str
            Artifact export path
        input_schema : Schema
            A schema with information about the inputs to this operator.
        output_schema : Schema
            A schema with information about the outputs of this operator.
        params : dict, optional
            Parameters dictionary of key, value pairs stored in exported config, by default None.
        node_id : int, optional
            The placement of the node in the graph (starts at 1), by default None.
        version : int, optional
            The version of the model, by default 1.

        Returns
        -------
        Ensemble_config: dict
            The config for the entire ensemble.
        Node_configs: list
            A list of individual configs for each step (operator) in graph.
        """

        params = params or {}

        node_name = f"{node_id}_{self.export_name}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(parents=True, exist_ok=True)

        config = model_config.ModelConfig(name=node_name, backend=backend, platform="op_runner")

        config.parameters["operator_names"].string_value = json.dumps([node_name])

        config.parameters[node_name].string_value = json.dumps(
            {
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                "input_dict": json.dumps(_schema_to_dict(input_schema)),
                "output_dict": json.dumps(_schema_to_dict(output_schema)),
                "params": json.dumps(params),
            }
        )

        for col_schema in input_schema.column_schemas.values():
            col_dims = compute_dims(col_schema)
            add_model_param(config.input, model_config.ModelInput, col_schema, col_dims)

        for col_schema in output_schema.column_schemas.values():
            col_dims = compute_dims(col_schema)
            add_model_param(config.output, model_config.ModelOutput, col_schema, col_dims)

        with open(os.path.join(node_export_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        os.makedirs(node_export_path, exist_ok=True)
        os.makedirs(os.path.join(node_export_path, str(version)), exist_ok=True)
        with importlib.resources.path(
            "merlin.systems.triton.models", "oprunner_model.py"
        ) as oprunner_model:
            copyfile(
                oprunner_model,
                os.path.join(node_export_path, str(version), "model.py"),
            )

        return config


def _schema_to_dict(schema: Schema) -> dict:
    # TODO: Write the conversion
    schema_dict = {}
    for col_name, col_schema in schema.column_schemas.items():
        schema_dict[col_name] = {
            "dtype": col_schema.dtype.name,
            "is_list": col_schema.is_list,
            "is_ragged": col_schema.is_ragged,
        }

    return schema_dict


def add_model_param(params, paramclass, col_schema, dims=None):
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=dims,
            )
        )
        params.append(
            paramclass(
                name=col_schema.name + "__lengths", data_type=model_config.TYPE_INT32, dims=dims
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )
