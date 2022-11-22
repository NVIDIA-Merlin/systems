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

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")  # noqa
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

from nvtabular.framework_utils.tensorflow import layers  # noqa


def create_tf_model(cat_columns: list, cat_mh_columns: list, embed_tbl_shapes: dict):
    inputs = {}  # tf.keras.Input placeholders for each feature to be used
    emb_layers = []  # output of all embedding layers, which will be concatenated
    for col in cat_columns:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.int64, shape=(1,))
    # Note that we need two input tensors for multi-hot categorical features
    for col in cat_mh_columns:
        inputs[col] = (
            tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,)),
            tf.keras.Input(name=f"{col}__lengths", dtype=tf.int64, shape=(1,)),
        )
    for col in cat_columns + cat_mh_columns:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, embed_tbl_shapes[col][0]
                ),  # Input dimension (vocab size)
                embed_tbl_shapes[col][1],  # Embedding output dimension
            )
        )
    emb_layer = layers.DenseFeatures(emb_layers)
    x_emb_output = emb_layer(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile("sgd", "binary_crossentropy")
    return model
