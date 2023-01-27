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
from unittest.mock import MagicMock, patch

import pytest

from merlin.schema import Schema
from merlin.systems.dag.ops.feast import QueryFeast
from merlin.systems.dag.runtimes.triton.ops.feast import QueryFeastTriton

feast = pytest.importorskip("feast")


def test_feast_config_round_trip(tmpdir):
    """
    This builds a QueryFeast op via the constructor, exports the config, and then builds another
    QueryFeast op via from_config to ensure that the constructor arguments match the original.
    """
    input_schema = Schema()
    output_schema = Schema()

    # We need to mock the FeatureStore constructor so it doesn't look for a real store, and the
    # QueryFeast constructor so that we can ensure that it gets called with the correct args.
    # These must be done in a context manager so we don't modify the classes globally and
    # affect other tests.
    with patch("feast.FeatureStore.__init__", MagicMock(return_value=None)):
        # Define the args & kwargs. We want to ensure the round-tripped version uses these same
        # arguments.
        args = [
            f"{tmpdir}",
            "entity_id",
            "entity_view",
            "entity_column",
            ["features"],
            ["mh_features"],
            input_schema,
            output_schema,
            True,  # include_id
            "prefix",  # output_prefix
        ]
        base_op = QueryFeast(*args)

        feast_op = QueryFeastTriton(base_op)

        created_config = feast_op.export(tmpdir + "/export_path/", input_schema, output_schema)
        created_config_dict = json.loads(created_config.parameters["queryfeasttriton"].string_value)

        with patch(
            "merlin.systems.dag.runtimes.triton.ops.feast.QueryFeastTriton.__init__",
            MagicMock(return_value=None),
        ), patch(
            "merlin.systems.dag.ops.feast.QueryFeast.__init__", MagicMock(return_value=None)
        ) as qf_init:

            # now mock the QueryFeast constructor so we can inspect its arguments.
            QueryFeastTriton.from_config(created_config_dict)
            qf_init.assert_called_with(*args)
