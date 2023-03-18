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

tf = pytest.importorskip("tensorflow")

triton = pytest.importorskip("merlin.systems.triton")

tritonclient = pytest.importorskip("tritonclient")
grpcclient = pytest.importorskip("tritonclient.grpc")

import merlin.models.tf as mm  # noqa
from merlin.datasets.synthetic import generate_data  # noqa
from merlin.io import Dataset  # noqa
from merlin.schema import Tags  # noqa
from merlin.systems.dag import Ensemble  # noqa
from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.triton.utils import run_ensemble_on_tritonserver  # noqa


def test_serve_tf_session_based_with_libtensorflow(tmpdir):

    # ===========================================
    # Generate training data
    # ===========================================

    train = generate_data("sequence-testing", num_rows=100)

    # ===========================================
    # Build and train the model
    # ===========================================

    seq_schema = train.schema.select_by_tag(Tags.SEQUENCE).select_by_tag(Tags.CATEGORICAL)

    target = train.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_last = mm.SequencePredictLast(schema=seq_schema, target=target)

    input_schema = seq_schema
    output_schema = seq_schema.select_by_name(target)

    train = Dataset(train.to_ddf(columns=input_schema.column_names).compute())
    train.schema = input_schema
    loader = mm.Loader(train, batch_size=16, shuffle=False)

    d_model = 48
    query_encoder = mm.Encoder(
        mm.InputBlockV2(
            input_schema,
            embeddings=mm.Embeddings(
                input_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
            ),
        ),
        mm.MLPBlock([d_model]),
        mm.GPT2Block(d_model=d_model, n_head=2, n_layer=2),
        tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    )

    model = mm.RetrievalModelV2(
        query=query_encoder,
        output=mm.ContrastiveOutput(output_schema, negative_samplers="in-batch"),
    )

    model.compile(metrics={})
    model.fit(loader, epochs=1, pre=predict_last)

    # ===========================================
    # Build a simple Ensemble graph
    # ===========================================
    tf_op = input_schema.column_names >> PredictTensorflow(
        model.query_encoder, input_schema, output_schema
    )

    ensemble = Ensemble(tf_op, input_schema)
    ens_config, node_configs = ensemble.export(str(tmpdir))

    # ===========================================
    # Create Request Data
    # ===========================================

    data = generate_data("sequence-testing", num_rows=1)
    request_df = data.compute()

    # ===========================================
    # Send request to Triton and check response
    # ===========================================
    response = run_ensemble_on_tritonserver(
        tmpdir, input_schema, request_df, ["output_1"], node_configs[0].name
    )

    assert response
    assert len(response["output_1"][0]) == d_model
