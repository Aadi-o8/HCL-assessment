import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

import api.main as api_main
import src.prediction as prediction
from api.schemas import FullCustomerFeatures, ReducedCustomerFeatures
from src.config import CATEGORIES, FEATURE_SETS, MODEL_FILES
from src.preprocessing import fit_preprocessor, transform


@pytest.fixture
def full_sample():
    return {
        **{name: values[0] for name, values in CATEGORIES.items()},
        "SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 60.0, "TotalCharges": 720.0,
    }


@pytest.fixture
def sample(full_sample):
    return {name: full_sample[name] for name in FEATURE_SETS[api_main.ACTIVE_FEATURE_SET]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "MODELS", tmp_path)
    monkeypatch.setattr(api_main, "REPORTS", tmp_path)
    (tmp_path / "best_model.json").write_text(json.dumps({"feature_set": api_main.ACTIVE_FEATURE_SET}))
    with TestClient(api_main.app) as client:
        yield client


def test_health_and_missing_artifacts(client, sample):
    assert client.get("/health").json() == {"status": "ok", "model_ready": False}
    catalog = client.get("/models").json()
    assert {item["id"] for item in catalog["models"]} == set(MODEL_FILES)
    assert all(not item["available"] for item in catalog["models"])
    assert catalog["required_features"] == FEATURE_SETS[api_main.ACTIVE_FEATURE_SET]
    assert client.post("/predict", json={"customer": sample}).status_code == 503
    assert client.post("/predict", json={
        "customer": sample, "model_id": "random_forest",
    }).status_code == 503
    assert client.get("/metrics").status_code == 503


def test_invalid_request(client, sample):
    for changes in ({"tenure": -1}, {"tenure": 0}, {"TotalCharges": 0}, {"InternetService": "No"},
                    {"SeniorCitizen": True}, {"MonthlyCharges": "60"}, {"Contract": "private-text"},
                    {"customerID": "private-text"}, {"TotalCharges": None}):
        response = client.post("/predict", json={"customer": {**sample, **changes}})
        assert response.status_code == 422
        assert "private-text" not in response.text
    assert client.post("/predict", json={"customer": {}}).status_code == 422
    assert client.post("/predict", json={
        "customer": sample, "model_id": "other",
    }).status_code == 422


def test_prediction_with_fitted_artifacts(client, sample, tmp_path):
    data = pd.DataFrame([sample, {**sample, "tenure": 60}])
    preprocessor = fit_preprocessor(data, FEATURE_SETS[api_main.ACTIVE_FEATURE_SET])
    model = LogisticRegression().fit(transform(preprocessor, data), [1, 0])
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    joblib.dump(model, tmp_path / "logistic_regression.joblib")
    (tmp_path / "best_model.json").write_text(json.dumps({
        "model_id": "logistic_regression", "feature_set": api_main.ACTIVE_FEATURE_SET,
    }))
    (tmp_path / "metrics.json").write_text(json.dumps({"selected_model": "logistic_regression"}))
    assert client.get("/health").json()["model_ready"]
    assert client.get("/models").json()["selected_model"] == "logistic_regression"
    assert client.get("/metrics").json()["selected_model"] == "logistic_regression"
    response = client.post("/predict", json={"customer": sample})
    assert response.status_code == 200
    expected = model.predict_proba(transform(preprocessor, data.iloc[:1]))[0, 1]
    assert response.json() == {
        "model_id": "logistic_regression", "churn_class": 1,
        "churn_score": pytest.approx(expected),
    }

@pytest.mark.parametrize("feature_set,schema", [
    ("full", FullCustomerFeatures), ("reduced", ReducedCustomerFeatures),
])
def test_schema_consistency(feature_set, schema, full_sample):
    fields = FEATURE_SETS[feature_set]
    assert set(schema.model_fields) == set(fields)
    customer = {name: full_sample[name] for name in fields}
    assert schema.model_validate(customer).tenure == 12
    no_internet = {
        **customer, "InternetService": "No",
        **{name: "No internet service" for name in fields
           if "No internet service" in CATEGORIES.get(name, ())},
    }
    assert schema.model_validate(no_internet).TechSupport == "No internet service"
    assert schema.model_validate({**no_internet, "tenure": 0, "TotalCharges": 0}).TotalCharges == 0
    for invalid in (
        {**no_internet, "OnlineSecurity": "Yes"},
        {**customer, "TechSupport": "No internet service"},
        {**customer, "MonthlyCharges": float("inf")},
        {**customer, "tenure": 0},
        {**customer, "TotalCharges": 0},
    ):
        with pytest.raises(ValidationError):
            schema.model_validate(invalid)