import pandas as pd

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from nvtabular import ColumnSelector


def test_softmax_output_dtype_keeps_input_dtype():
    # We expect the method to:
    # * change the output name to `ordered_ids`
    # * change is_list to True
    # * change is_ragged to True
    # * Not change the output dtype

    s = SoftmaxSampling("rel_col", _input_col="input_col")

    input_col_schema = Schema(
        [
            ColumnSchema(name="rel_col"),
            ColumnSchema(name="input_col", dtype=pd.StringDtype, is_list=False, is_ragged=False),
        ]
    )

    expected = Schema(
        [ColumnSchema(name="ordered_ids", dtype=pd.StringDtype, is_list=True, is_ragged=True)]
    )
    actual = s.compute_output_schema(input_col_schema, ColumnSelector(["ordered_ids"]))

    assert actual == expected
