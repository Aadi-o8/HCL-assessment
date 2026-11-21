import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import FEATURE_SETS, MODEL_FILES, MODELS, THRESHOLD
from src.evaluation import churn_scores
from src.neural_network import tensorflow
from src.preprocessing import transform


class ModelNotReady(RuntimeError):
    pass


def deployment_feature_set():
    try:
        metadata = json.loads((MODELS / "best_model.json").read_text(encoding="utf-8"))
        feature_set = metadata.get("feature_set", "full")
        if feature_set in FEATURE_SETS and metadata.get("raw_features", FEATURE_SETS[feature_set]) == FEATURE_SETS[feature_set]:
            return feature_set
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return "full"


def selected_model():
    try:
        model_id = json.loads((MODELS / "best_model.json").read_text(encoding="utf-8"))["model_id"]
        if model_id in MODEL_FILES:
            return model_id
    except (OSError, ValueError, KeyError, TypeError):
        pass
    raise ModelNotReady("Train the project to create deployment artifacts.")


def available_models():
    if not has_artifact("preprocessor.joblib"):
        return []
    return [name for name, filename in MODEL_FILES.items() if has_artifact(filename)]


def has_artifact(filename):
    try:
        return (MODELS / filename).stat().st_size > 0
    except OSError:
        return False


@lru_cache(maxsize=5)
def load_artifact(path, modified_ns):
    # The timestamp keys the cache so retraining can replace an artifact.
    if path.suffix == ".keras":
        return tensorflow().keras.models.load_model(path, compile=False)
    return joblib.load(path)


def predict(raw_features, model_id=None):
    model_id = selected_model() if model_id is None else model_id
    if model_id not in MODEL_FILES:
        raise ValueError("Unknown model ID.")
    preprocessor_path = MODELS / "preprocessor.joblib"
    model_path = MODELS / MODEL_FILES[model_id]
    try:
        preprocessor = load_artifact(preprocessor_path, preprocessor_path.stat().st_mtime_ns)
        model = load_artifact(model_path, model_path.stat().st_mtime_ns)
    except Exception as error:
        raise ModelNotReady("Model artifacts are unavailable. Run training and retry.") from error
    expected = FEATURE_SETS[deployment_feature_set()]
    if list(preprocessor.feature_names_in_) != expected or set(raw_features) != set(expected):
        raise ModelNotReady("Prediction fields and the saved preprocessor differ.")
    features = transform(preprocessor, pd.DataFrame([raw_features]))
    dimension = model.input_shape[-1] if model_path.suffix == ".keras" else model.n_features_in_
    if features.shape[1] != dimension:
        raise ModelNotReady("The saved preprocessor and classifier disagree on input dimensions.")
    score = float(churn_scores(model, model_id, features)[0])
    return {"model_id": model_id, "churn_class": int(score >= THRESHOLD), "churn_score": score}
