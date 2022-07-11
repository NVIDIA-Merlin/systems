from merlin.datasets.synthetic import generate_data
from merlin.models.implicit import BayesianPersonalizedRanking
from merlin.systems.dag.ops.implicit import PredictImplicit


def test_predict_implcit(tmpdir):
    dataset = generate_data("movielens-100k", num_rows=100)
    dataset.schema = dataset.schema.excluding_by_name("rating_binary")

    model = BayesianPersonalizedRanking()
    model.fit(dataset)

    op = PredictImplicit(model)

    op.export(tmpdir, None, None)
