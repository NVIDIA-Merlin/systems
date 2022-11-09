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
from typing import Dict, Optional

import numpy as np

from merlin.core.protocols import SeriesLike, Transformable


class Column(SeriesLike):
    """
    A simple wrapper around an array of values
    """

    def __init__(self, values):
        super().__init__()

        if isinstance(values, Column):
            raise TypeError("doubly nested columns")

        self.values = values
        self.dtype = values.dtype

    def __getitem__(self, index):
        return self.values[index]

    def __eq__(self, other):
        return all(self.values == other.values) and self.dtype == other.dtype

    def __len__(self):
        return len(self.values)

    @property
    def shape(self):
        return self.values.shape


class DictArray(Transformable):
    """
    A simple dataframe-like wrapper around a dictionary of values
    """

    def __init__(self, values: Optional[Dict] = None):
        super().__init__()
        values = values or {}

        self._columns = {key: _make_column(value) for key, value in values.items()}

    @property
    def columns(self):
        return list(self._columns.keys())

    @property
    def dtypes(self):
        return {key: value.dtype for key, value in self._columns.items()}

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

    def __eq__(self, other):
        return self._columns == other.values and self.dtypes == other.dtypes

    def __setitem__(self, key, value):
        self._columns[key] = _make_column(value)

    def __getitem__(self, key):
        if isinstance(key, list):
            return DictArray({k: self._columns[k] for k in key})
        else:
            return self._columns[key]

    def __delitem__(self, key):
        del self._columns[key]

    def keys(self):
        """
        Shortcut to get the dictionary keys
        """
        return self._columns.keys()

    def items(self):
        """
        Shortcut to get the dictionary items
        """
        return self._columns.items()

    def values(self):
        """
        Shortcut to get the dictionary values
        """
        return self._columns.values()

    def update(self, other):
        """
        Shortcut to update the dictionary items
        """
        self._columns.update(other)

    def copy(self):
        """
        Create a new DictArray with the same data and dtypes
        """
        return DictArray(self._columns.copy())


def _make_column(value):
    value = np.array(value) if isinstance(value, list) else value
    column = Column(value) if not isinstance(value, Column) else value
    return column
