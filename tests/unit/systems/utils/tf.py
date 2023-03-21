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


class ExampleModel(tf.keras.Model):
    def __init__(self, cat_columns, cat_mh_columns, embed_tbl_shapes):
        super(ExampleModel, self).__init__()

        self.cat_columns = cat_columns
        self.cat_mh_columns = cat_mh_columns
        self.embed_tbl_shapes = embed_tbl_shapes

        self.emb_layers = []  # output of all embedding layers, which will be concatenated
        for col in self.cat_columns + self.cat_mh_columns:
            self.emb_layers.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_identity(
                        col, self.embed_tbl_shapes[col][0]
                    ),  # Input dimension (vocab size)
                    self.embed_tbl_shapes[col][1],  # Embedding output dimension
                )
            )
        self.emb_layer = layers.DenseFeatures(self.emb_layers)
        self.dense_layer = tf.keras.layers.Dense(128, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="output")

    def call(self, inputs):
        reshaped_inputs = {}
        for input_name, input_ in inputs.items():
            reshaped_inputs[input_name] = tf.reshape(input_, (-1, 1))

        x_emb_output = self.emb_layer(reshaped_inputs)
        x = self.dense_layer(x_emb_output)
        x = self.output_layer(x)
        return {"predictions": x}


def create_tf_model(cat_columns: list, cat_mh_columns: list, embed_tbl_shapes: dict):

    model = ExampleModel(cat_columns, cat_columns, embed_tbl_shapes)
    model.compile("sgd", "binary_crossentropy")
    example_input = tf.constant([1, 2, 3], dtype=tf.int64)
    cat_column_data = {cat_col: example_input for cat_col in cat_columns}
    model(cat_column_data)
    return model
