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

from merlin.dag import BaseOperator  # noqa
from merlin.dag.selector import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.node import InferenceNode  # noqa
from merlin.systems.triton.export import _convert_dtype  # noqa


class InferenceDataFrame:
    def __init__(self, tensors=None):
        """
        This is a dictionary that has a set of key (column name) and value (tensor)

        Parameters
        ----------
        tensors : Dictionary, optional
            A dictionary consisting of column name (key), tensor vector (value) , by default None
        """
        self.tensors = tensors or {}

    def __getitem__(self, col_items):
        if isinstance(col_items, list):
            results = {name: self.tensors[name] for name in col_items}
            return InferenceDataFrame(results)
        else:
            return self.tensors[col_items]

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        for name, tensor in self.tensors.items():
            yield name, tensor

    def __repr__(self):
        dict_rep = {}
        for k, v in self.tensors.items():
            dict_rep[k] = v
        return str(dict_rep)


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

    @abstractmethod
    def export(
        self,
        path: str,
        input_schema: Schema,
        output_schema: Schema,
        params: dict = None,
        node_id: int = None,
        version: int = 1,
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

    @property
    def scalar_shape(self):
        return [1]


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
    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
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

        for col_name, col_dict in _schema_to_dict(input_schema).items():
            config.input.append(
                model_config.ModelInput(
                    name=col_name, data_type=_convert_dtype(col_dict["dtype"]), dims=[-1, -1]
                )
            )

        for col_name, col_dict in _schema_to_dict(output_schema).items():
            # this assumes the list columns are 1D tensors both for cats and conts
            config.output.append(
                model_config.ModelOutput(
                    name=col_name,
                    data_type=_convert_dtype(col_dict["dtype"]),
                    dims=[-1, -1],
                )
            )

        with open(os.path.join(node_export_path, "config.pbtxt"), "w", encoding="utf-8") as o:
            text_format.PrintMessage(config, o)

        os.makedirs(node_export_path, exist_ok=True)
        os.makedirs(os.path.join(node_export_path, str(version)), exist_ok=True)
        copyfile(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "triton", "models", "oprunner_model.py"
            ),
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
                name=col_schema.name + "__nnzs", data_type=model_config.TYPE_INT32, dims=dims
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )
