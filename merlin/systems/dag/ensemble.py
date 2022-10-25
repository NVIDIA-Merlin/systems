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
import sys
import time
import warnings

import cloudpickle
import fsspec

from merlin.core.protocols import Transformable
from merlin.dag import Graph
from merlin.systems.dag.runtimes.triton import TritonEnsembleRuntime


class Ensemble:
    """
    Class that represents an entire ensemble consisting of multiple models that
    run sequentially in tritonserver initiated by an inference request.
    """

    def __init__(self, ops, schema, label_columns=None):
        """Construct a systems ensemble.

        Parameters
        ----------
        ops : InferenceNode
            An inference node that represents the chain of operators for the ensemble.
        schema : Schema
            The schema of the input data.
        label_columns : List[str], optional
            List of strings representing label columns, by default None
        """
        self.graph = Graph(ops)
        self.graph.construct_schema(schema)
        self.label_columns = label_columns or []

    @property
    def input_schema(self):
        return self.graph.input_schema

    @property
    def output_schema(self):
        return self.graph.output_schema

    def transform(self, transformable: Transformable, runtime=None):
        """Delegate transformation of input data to the executor with
        the necessary ensemble graph. This will traverse each node of
        the graph and mutate the input according to the operator in each
        node.

        Parameters
        ----------
        transformable : Transformable
            Input data to the graph (DataFrame or DictArray).
        runtime : Runtime, optional
            The graph runtime to use to transform the inputs, by default None

        Returns
        -------
        Transformable
            transformed data
        """
        runtime = runtime or TritonEnsembleRuntime()
        return runtime.transform(self.graph, transformable)

    def save(self, path):
        """Save this ensemble to disk

        Parameters
        ----------
        path: str
            The path to save the ensemble to
        """
        fs = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)

        # TODO: Include the systems version in the metadata file below

        # generate a file of all versions used to generate this bundle
        with fs.open(fs.sep.join([path, "metadata.json"]), "w") as o:
            json.dump(
                {
                    "versions": {
                        "python": sys.version,
                    },
                    "generated_timestamp": int(time.time()),
                },
                o,
            )

        # dump out the full workflow (graph/stats/operators etc) using cloudpickle
        with fs.open(fs.sep.join([path, "ensemble.pkl"]), "wb") as o:
            cloudpickle.dump(self, o)

    @classmethod
    def load(cls, path) -> "Ensemble":
        """Load up a saved ensemble object from disk

        Parameters
        ----------
        path: str
            The path to load the ensemble from

        Returns
        -------
        Ensemble
            The ensemble loaded from disk
        """
        fs = fsspec.get_fs_token_paths(path)[0]

        # check version information from the metadata blob, and warn if we have a mismatch
        meta = json.load(fs.open(fs.sep.join([path, "metadata.json"])))

        def parse_version(version):
            return version.split(".")[:2]

        def check_version(stored, current, name):
            if parse_version(stored) != parse_version(current):
                warnings.warn(
                    f"Loading workflow generated with {name} version {stored} "
                    f"- but we are running {name} {current}. This might cause issues"
                )

        # make sure we don't have any major/minor version conflicts between the stored worklflow
        # and the current environment
        versions = meta["versions"]
        check_version(versions["python"], sys.version, "python")

        ensemble = cloudpickle.load(fs.open(fs.sep.join([path, "ensemble.pkl"]), "rb"))

        return ensemble

    def export(self, export_path, runtime=None, **kwargs):
        """
        Write out an ensemble model configuration directory. The exported
        ensemble is designed for use with Triton Inference Server.
        """
        runtime = runtime or TritonEnsembleRuntime()
        return runtime.export(self, export_path, **kwargs)
