# Copyright (c) 2021, NVIDIA CORPORATION.
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
import warnings

warnings.warn(
    "The `merlin.legacy.inference` module is being replaced by the Merlin Systems library. "
    "Support for importing from `merlin.legacy.inference` is deprecated, "
    "and will be removed in a future version. Please consider using the "
    "models and layers from Merlin Systems instead.",
    DeprecationWarning,
)