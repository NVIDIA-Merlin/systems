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
# pylint: disable=unused-import

try:
    import xgboost
except ImportError:
    xgboost = None
try:
    import sklearn.ensemble as sklearn_ensemble
except ImportError:
    sklearn_ensemble = None
try:
    import lightgbm
except ImportError:
    lightgbm = None
try:
    import cuml.ensemble as cuml_ensemble
except ImportError:
    cuml_ensemble = None
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None
