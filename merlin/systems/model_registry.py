from abc import ABC, abstractmethod

import requests


class ModelRegistry(ABC):
    """
    The ModelRegistry class is used to find model paths that will be imported into an
    InferenceOperator.

    To implement your own ModelRegistry subclass, the only method that must be implemented is
    `get_artifact_uri`, which must return a string indicating the model's export path.

    ```python
    PredictTensorflow.from_model_registry(
        MyModelRegistry("model_name", "model_version")
    )
    ```
    """

    @abstractmethod
    def get_artifact_uri(self) -> str:
        """
        This returns the URI of the model artifact.
        """


class MLFlowModelRegistry(ModelRegistry):
    def __init__(self, name: str, version: str, tracking_uri: str):
        """
        Fetches the model path from an mlflow model registry.

        Note that this will return a relative path if you did not configure your mlflow
        experiment's `artifact_location` to be an absolute path.

        Parameters
        ----------
        name : str
            Name of the model in the mlflow registry.
        version : str
            Version of the model to use.
        tracking_uri : str
            Base URI of the mlflow tracking server. If running locally, this would likely be
            http://localhost:5000
        """
        self.name = name
        self.version = version
        self.tracking_uri = tracking_uri.rstrip("/")

    def get_artifact_uri(self) -> str:
        mv = requests.get(
            f"{self.tracking_uri}/ajax-api/2.0/preview/mlflow/model-versions/get-download-uri",
            params={"name": self.name, "version": self.version},
        )

        if mv.status_code != 200:
            raise ValueError(
                f"Could not find a Model Version for model {self.name} with version {self.version}."
            )
        model_path = mv.json()["artifact_uri"]
        return model_path
