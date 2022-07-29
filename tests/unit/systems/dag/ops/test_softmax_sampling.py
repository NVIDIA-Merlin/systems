import numpy as np
import pandas as pd
import pytest

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceDataFrame
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from nvtabular import ColumnSelector


def test_softmax_output_dtype_keeps_input_dtype():
    # We expect the method to not change the output dtype

    s = SoftmaxSampling("rel_col", _input_cols=["input_col"])

    input_col_schema = ColumnSchema(
        name="input_col", dtype=pd.StringDtype, is_list=False, is_ragged=False
    )
    input_schema = Schema([ColumnSchema(name="rel_col"), input_col_schema])

    actual = s.compute_output_schema(input_schema, ColumnSelector(["input_col"]))

    assert actual == Schema([input_col_schema])


@pytest.mark.parametrize("dtype", [np.float32, pd.StringDtype])
@pytest.mark.parametrize(
    "input_cols", [["input_col1"], ["input_col2"], ["input_col1", "input_col2"]]
)
def test_softmax_output_dtype__with_multiple_inputs_keeps_input_dtype(input_cols, dtype):
    # We expect the method to not change the output dtype

    s = SoftmaxSampling("rel_col", _input_cols=input_cols)

    input_col_schema = [
        ColumnSchema(name=input_col, dtype=dtype, is_list=False, is_ragged=False)
        for input_col in input_cols
    ]

    input_schema = Schema([ColumnSchema(name="rel_col")] + input_col_schema)

    actual = s.compute_output_schema(input_schema, ColumnSelector(input_cols))
    assert actual == Schema(input_col_schema)


@pytest.mark.parametrize(
    "input_cols", [["input_col1"], ["input_col2"], ["input_col1", "input_col2"]]
)
def test_softmax_ordering(input_cols):
    s = SoftmaxSampling("rel_col", _input_cols=input_cols)
    tensors = {col: np.random.uniform(0, 1, 3) for col in input_cols}
    tensors.update({"rel_col": np.array([1, 0.1, 0.01])})
    df = InferenceDataFrame(tensors)

    transformed = s.transform(df)
    assert sorted(transformed.tensors.keys()) == sorted(input_cols)
