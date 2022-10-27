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
from merlin.core.protocols import Transformable
from merlin.dag import Graph
from merlin.dag.executors import LocalExecutor


class Runtime:
    """A Systems Graph Runtime.

    This class can be used as a base class for custom runtimes.
    """

    def __init__(self, executor=None):
        """Construct a Runtime.

        Parameters
        ----------
        executor : Executor, optional
            The Graph Executor to use to use for the transform, by default None
        """
        self.executor = executor or LocalExecutor()
        self.op_table = {}

    def transform(self, graph: Graph, transformable: Transformable):
        """Run the graph with the input data.

        Parameters
        ----------
        graph : Graph
            Graph of nodes container operator chains for data manipulation.
        transformable : Transformable
            Input data to transform in graph.

        Returns
        -------
        Transformable
            Input data after it has been transformed via graph.
        """
        return self.executor.transform(transformable, [graph.output_node])

    def export(self):
        """Optional method.
        Implemented for runtimes that require an exported artifact to transform.
        """
        raise NotImplementedError
