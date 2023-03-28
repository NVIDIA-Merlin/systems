from abc import abstractmethod

from merlin.core.protocols import Transformable  # noqa
from merlin.dag import BaseOperator  # noqa
from merlin.dag.selector import ColumnSelector  # noqa
from merlin.schema import Schema  # noqa
from merlin.systems.dag.node import InferenceNode  # noqa
from merlin.systems.model_registry import ModelRegistry


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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a Transform function."
            "Please create one in the operator class, where you inherited "
            "from the base operator."
        )

    def load_artifacts(self, artifact_path: str) -> None:
        """Load artifacts from disk required for operator function.

        Parameters
        ----------
        artifact_path : str
            The path where artifacts are loaded from
        """

    def save_artifacts(self, artifact_path: str) -> None:
        """Save artifacts required to be reload operator state from disk

        Parameters
        ----------
        artifact_path : str
            The path where artifacts are to be saved
        """

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
        model_config: dict
            The config for the exported operator (Triton model).
        """
        raise NotImplementedError(
            "Exporting an operator to run in a particular context (i.e. Triton) "
            "only makes sense when a runtime is specified. To select an "
            f"operator for the appropriate runtime, replace {self.__class__.__name__} "
            f"with a runtime-specific operator class, possibly {self.__class__.__name__}Triton"
        )

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
