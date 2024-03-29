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
import warnings

from . import _version

# suppress some warnings with cudf warning about column ordering with dlpack
# and numba warning about deprecated environment variables
warnings.filterwarnings("ignore", module="cudf.io.dlpack")
warnings.filterwarnings("ignore", module="numba.cuda.envvars")

# cudf warns about column ordering with dlpack methods, ignore it
warnings.filterwarnings("ignore", module="cudf.io.dlpack")


__version__ = _version.get_versions()["version"]
