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
import pytest

torch = pytest.importorskip("torch")  # noqa

from nvtabular.framework_utils.torch.models import Model  # noqa


def create_pytorch_model(cat_columns: list, cat_mh_columns: list, embed_tbl_shapes: dict):
    single_hot = {k: v for k, v in embed_tbl_shapes.items() if k in cat_columns}
    multi_hot = {k: v for k, v in embed_tbl_shapes.items() if k in cat_mh_columns}
    model = Model(
        embedding_table_shapes=(single_hot, multi_hot),
        num_continuous=0,
        emb_dropout=0.0,
        layer_hidden_dims=[128, 128, 128],
        layer_dropout_rates=[0.0, 0.0, 0.0],
    ).to("cuda")
    return model
