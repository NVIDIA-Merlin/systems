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
import numpy as np

from merlin.core.dispatch import get_lib
from merlin.systems.dag.dictarray import DictArray


def test_column_scalar_list():
    values = np.array([[1], [2], [3]])
    dict_array = DictArray({"col": values})
    df_from_dictarray = dict_array.to_df()
    df_created = get_lib().DataFrame({"col": values.tolist()})
    assert df_from_dictarray.to_string() == df_created.to_string()


def test_single_value_list_column():
    request_features = {
        "user_id": np.array([1]),
    }
    request_data = DictArray(request_features)
    assert request_data["user_id"].values() == np.array([1])


def test_multiple_arrays_to_dict():
    ids = np.array(
        [[12, 87, 90, 64, 32, 53, 85, 75, 51, 14], [12, 87, 90, 64, 32, 53, 85, 75, 51, 14]],
        dtype=np.int32,
    )
    scores = np.array(
        [
            [
                0.08853003,
                0.08152322,
                0.0739782,
                0.07222077,
                0.05578036,
                0.03810744,
                0.03461488,
                0.0334378,
                0.02731128,
                0.02388261,
            ],
            [
                0.08998033,
                0.08277725,
                0.07515433,
                0.07345057,
                0.05647209,
                0.03882845,
                0.03512128,
                0.03392398,
                0.02807988,
                0.02420903,
            ],
        ],
        dtype=np.float32,
    )

    dict_array = DictArray({"ids": ids.astype(np.int64), "scores": scores.astype(np.float64)})

    assert np.array_equal(dict_array["ids"].values, ids.astype(np.int64))
    assert np.array_equal(dict_array["scores"].values, scores.astype(np.float64))
