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
    def __init__(self, executor=None):
        self.executor = executor or LocalExecutor()

    def transform(self, graph: Graph, transformable: Transformable):
        return self.executor.transform(transformable, [graph.output_node])

    def export(self):
        raise NotImplementedError
