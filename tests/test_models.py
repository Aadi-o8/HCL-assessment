import joblib
import numpy as np
import pytest

from src.config import MODEL_FILES
from src.evaluation import classification_metrics, churn_scores, select_model
from src.neural_network import fit_mlp, tensorflow
from src.traditional_ml import train_traditional
from train import compare_features


@pytest.fixture
def small_data():
    features = np.random.default_rng(42).normal(size=(64, 5)).astype(np.float32)
    labels = np.tile([0, 1], 32)
    return features[:48], labels[:48], features[48:], labels[48:]


def test_traditional_training_and_serialization(small_data, tmp_path):
    x_train, y_train, x_validation, y_validation = small_data
    models = train_traditional(x_train, y_train, x_validation, y_validation, quick=True)
    assert set(models) == {"logistic_regression", "random_forest"}
    for name, model in models.items():
        path = tmp_path / f"{name}.joblib"
        joblib.dump(model, path)
        scores = churn_scores(model, name, x_validation)
        assert scores.shape == (16,)
        assert ((scores >= 0) & (scores <= 1)).all()
        np.testing.assert_allclose(scores, churn_scores(joblib.load(path), name, x_validation))


@pytest.mark.parametrize("name,configuration", [
    ("basic_mlp", ((32,), 0.0, 0.001)),
    ("optimized_mlp", ((64, 32), 0.3, 0.001)),
])
def test_mlp_training_and_serialization(small_data, tmp_path, name, configuration):
    model, history, metrics = fit_mlp(*small_data, configuration, epochs=1)
    assert model.input_shape == (None, 5)
    assert model.output_shape == (None, 1)
    assert history["best_epoch"] == 1
    assert metrics["roc_auc"] is not None
    scores = churn_scores(model, name, small_data[2])
    assert scores.shape == (16,)
    assert ((scores >= 0) & (scores <= 1)).all()
    path = tmp_path / f"{name}.keras"
    model.save(path)
    loaded = tensorflow().keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(scores, churn_scores(loaded, name, small_data[2]))


def test_metrics_and_tie_break():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.5, 0.4, 0.9])
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert classification_metrics([0, 0], [0.1, 0.2])["roc_auc"] is None
    assert select_model({"b": metrics, "a": metrics}) == "a"

def test_feature_selection_rejects_mean_f1_or_best_auc_drop():
    full = {name: {"roc_auc": 0.84, "f1": 0.60} for name in MODEL_FILES}
    comparable = {name: {"roc_auc": 0.835, "f1": 0.58} for name in MODEL_FILES}
    low_f1 = {name: {"roc_auc": 0.84, "f1": 0.56} for name in MODEL_FILES}
    assert compare_features(full, comparable)[0] == "reduced"
    assert compare_features(full, low_f1)[0] == "full"
    # An isolated strong full-model result can be hidden by the average.
    full["basic_mlp"]["roc_auc"] = 0.85
    assert compare_features(full, comparable)[0] == "full"
