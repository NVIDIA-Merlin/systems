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
import json

from merlin.dag import ColumnSelector


class OperatorRunner:
    """Runner for collection of operators in one triton model."""

    def __init__(
        self,
        config,
        *,
        model_repository="./",
        model_version=1,
        model_name=None,
    ):
        """Instantiate an OperatorRunner"""
        operator_names = self.fetch_json_param(config, "operator_names")
        op_configs = [self.fetch_json_param(config, op_name) for op_name in operator_names]

        self.operators = []
        for op_config in op_configs:
            module_name = op_config["module_name"]
            class_name = op_config["class_name"]

            op_module = importlib.import_module(module_name)
            op_class = getattr(op_module, class_name)

            operator = op_class.from_config(
                op_config,
                model_repository=model_repository,
                model_name=model_name,
                model_version=model_version,
            )
            self.operators.append(operator)

    def execute(self, tensors):
        """Run transform on multiple operators"""

        selector = ColumnSelector("*")
        for operator in self.operators:
            input_type = type(tensors)
            tensors = operator.transform(selector, tensors)
            if isinstance(tensors, dict):
                tensors = input_type(tensors)

        return tensors

    def fetch_json_param(self, model_config, param_name):
        """Extract JSON value from model config parameters"""
        string_value = model_config["parameters"][param_name]["string_value"]
        return json.loads(string_value)
