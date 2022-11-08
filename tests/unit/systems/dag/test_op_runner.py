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
import os
from unittest import mock

import numpy as np
import pytest
from google.protobuf.json_format import MessageToDict

import nvtabular as nvt
import nvtabular.ops as wf_ops
from merlin.dag import Graph
from merlin.schema import Tags
from merlin.systems.dag import Column, DictArray
from tests.unit.systems.utils.ops import PlusTwoOp

op_runner = pytest.importorskip("merlin.systems.dag.op_runner")
inf_op = pytest.importorskip("merlin.systems.dag.ops.operator")


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_config(tmpdir, dataset, engine):
    input_columns = ["x", "y", "id"]

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit(dataset)
    workflow.save(str(tmpdir))

    repository = "repository_path/"
    version = 1
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["PlusTwoOp_1"])},
            "PlusTwoOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
        }
    }

    runner = op_runner.OperatorRunner(config, model_repository=repository, model_version=version)

    loaded_op = runner.operators[0]
    assert isinstance(loaded_op, PlusTwoOp)


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_multiple_ops_same(tmpdir, dataset, engine):
    # NVT
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    repository = "repository_path/"
    version = 1
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["PlusTwoOp_1", "PlusTwoOp_2"])},
            "PlusTwoOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
            "PlusTwoOp_2": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
        }
    }

    runner = op_runner.OperatorRunner(
        config,
        model_repository=repository,
        model_version=version,
    )

    assert len(runner.operators) == 2

    for idx, loaded_op in enumerate(runner.operators):
        assert isinstance(loaded_op, PlusTwoOp)


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_multiple_ops_same_execute(tmpdir, dataset, engine):
    # NVT
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    repository = "repository_path/"
    version = 1
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["PlusTwoOp_1", "PlusTwoOp_2"])},
            "PlusTwoOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
            "PlusTwoOp_2": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
        }
    }

    runner = op_runner.OperatorRunner(
        config,
        model_repository=repository,
        model_version=version,
    )

    inputs = {}
    for col_name in schema.column_names:
        inputs[col_name] = Column(np.random.randint(10, size=(10,)))

    outputs = runner.execute(DictArray(inputs))

    assert outputs["x_plus_2_plus_2"] == Column(inputs["x"].values + 4)


@pytest.mark.parametrize("engine", ["parquet"])
@mock.patch.object(PlusTwoOp, "from_config", side_effect=PlusTwoOp.from_config)
def test_op_runner_single_node_export(mock_from_config, tmpdir, dataset, engine):
    # assert against produced config
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    inputs = ["x", "y"]

    node = inputs >> PlusTwoOp()

    graph = Graph(node)
    graph.construct_schema(dataset.schema)

    config = node.export(tmpdir)

    file_path = os.path.join(str(tmpdir), node.export_name, "config.pbtxt")

    assert os.path.exists(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        config_file = f.read()
    assert config_file == str(config)
    assert len(config.input) == len(inputs)
    assert len(config.output) == len(inputs)
    for idx, conf in enumerate(config.output):
        assert conf.name == inputs[idx] + "_plus_2"

    runner = op_runner.OperatorRunner(
        MessageToDict(
            config, preserving_proto_field_name=True, including_default_value_fields=True
        ),
        model_repository=str(tmpdir),
        model_name=config.name,
        model_version="1",
    )
    inputs = DictArray({"x": np.array([1]), "y": np.array([5])}, {"x": np.int32, "y": np.int32})
    outputs = runner.execute(inputs)

    assert outputs["x_plus_2"] == Column(np.array([3]))
    assert outputs["y_plus_2"] == Column(np.array([7]))

    assert mock_from_config.call_count == 1
    assert mock_from_config.call_args.kwargs == {
        "model_repository": str(tmpdir),
        "model_name": config.name,
        "model_version": "1",
    }
