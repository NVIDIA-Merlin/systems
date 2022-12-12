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
from enum import Enum
from typing import Dict, Optional

import numpy as np

from merlin.core.dispatch import get_lib
from merlin.core.protocols import SeriesLike

try:
    import cupy
except ImportError:
    cupy = None


class Device(Enum):
    CPU = 0
    GPU = 1


class Column(SeriesLike):
    """
    A simple wrapper around an array of values. This has an API that's just similar enough to
    Pandas and cuDF series to be relatively interchangeable from the perspective of the Merlin DAG,
    but no more. As more methods get added to this class, it gets closer and closer to actually
    *being* a Pandas/cuDF Series (at which point there's no advantage to using this.) So: keep
    this class as small as possible.
    """

    def __init__(self, values, row_lengths=None):
        super().__init__()

        self.values = _make_array(values)
        self.row_lengths = _make_array(row_lengths)
        self.dtype = self.values.dtype

        if isinstance(self.values, np.ndarray):
            self._device = Device.CPU
        elif cupy and isinstance(self.values, cupy.ndarray):
            self._device = Device.GPU
        else:
            raise TypeError(
                "Column only supports values of type numpy.ndarray or cupy.ndarray. "
                f"To use another type (like {type(values)}), convert to one of these types first."
            )

    def cpu(self):
        """
        Move the data for this column to host (CPU) memory

        Returns
        -------
        Column
            Same column, same data but now definitely in CPU memory
        """
        self.device = Device.CPU
        return self

    def gpu(self):
        """
        Move the data for this column to device (GPU) memory

        Returns
        -------
        Column
            Same column, same data but now definitely in GPU memory
        """
        self.device = Device.GPU
        return self

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if not cupy:
            raise ValueError(
                "Unable to move Column data between CPU and GPU without Cupy installed."
            )

        if device == "cpu":
            device = Device.CPU
        elif device == "gpu":
            device = Device.GPU

        # CPU to GPU
        if self._device == Device.CPU and device == Device.GPU:
            self._device_move(cupy.asarray)
        # GPU to CPU
        elif self._device == Device.GPU and device == Device.CPU:
            self._device_move(cupy.asnumpy)
        # Nothing to do
        else:
            return self

    def _device_move(self, fn):
        self.values = fn(self.values)
        if self.row_lengths:
            self.row_lengths = fn(self.row_lengths)

    def __getitem__(self, index):
        if self.row_lengths:
            start = self._array_lib.cumsum(self.row_lengths[:index])
            end = start + self.row_lengths[index] - 1
            return self.values[start:end]
        else:
            return self.values[index]

    def __eq__(self, other):
        values_eq = all(self.values == other.values) and self.dtype == other.dtype
        if self.row_lengths:
            return values_eq and all(self.row_lengths == other.row_lengths)
        else:
            return values_eq

    def __len__(self):
        if self.row_lengths:
            return len(self.row_lengths)
        else:
            return len(self.values)

    @property
    def shape(self):
        if self.row_lengths:
            dim = self.row_lengths[0] if self.is_ragged else None
            return (len(self), dim)
        else:
            return self.values.shape

    @property
    def is_list(self):
        return (
            len(self.values.shape) > 1
            or self.row_lengths is not None
            or isinstance(self.values[0], np.ndarray)
        )

    @property
    def is_ragged(self):
        return self.row_lengths and any(self.row_lengths != self.row_lengths[0])

    @property
    def _array_lib(self):
        return cupy if cupy and self.device == Device.GPU else np


class DictArray:
    """
    A simple dataframe-like wrapper around a dictionary of values. Matches the Transformable
    protocol for (limited) interchangeability with actual dataframes in Merlin DAGs.
    """

    def __init__(self, values: Optional[Dict] = None):
        super().__init__()
        values = values or {}

        self._columns = {key: _make_column(value) for key, value in values.items()}

    @property
    def columns(self):
        return list(self._columns.keys())

    # TODO: Make this a property once the underlying Transformable protocol has been adjusted
    def dtypes(self):
        return {key: value.dtype for key, value in self._columns.items()}

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

    def __eq__(self, other):
        return self.values() == other.values() and self.dtypes() == other.dtypes()

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

    def to_df(self):
        """
        Create a DataFrame from the DictArray
        """
        df = get_lib().DataFrame()
        for col in self.columns:
            df[col] = get_lib().Series(self[col])
        return df

    @classmethod
    def from_df(cls, df):
        """
        Create a DictArray from a DataFrame
        """
        array_dict = {}
        for col in df.columns:
            array_dict[col] = df[col].to_numpy()
        return cls(array_dict)


def _array_lib():
    """Dispatch to the appropriate library (cupy or numpy) for the current environment"""
    return cupy if cupy else np


def _make_column(value):
    if isinstance(value, tuple):
        values, row_lengths = value
        return Column(values, row_lengths=row_lengths)
    else:
        column = Column(value) if not isinstance(value, Column) else value
        return column


def _make_array(value):
    return _array_lib().array(value) if isinstance(value, list) else value
