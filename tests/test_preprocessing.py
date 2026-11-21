import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORIES, CUSTOMER_ID, REDUCED_FEATURES, TARGET
from src.ingestion import clean_data, split_data
from src.preprocessing import fit_preprocessor, transform


def raw_rows(count=40):
    template = {name: options[0] for name, options in CATEGORIES.items()}
    return pd.DataFrame([
        {**template, CUSTOMER_ID: f"id-{index}", TARGET: " Yes " if index % 2 else " No ",
         "SeniorCitizen": 0, "tenure": index, "MonthlyCharges": 20 + index,
         "TotalCharges": " " if index < 2 else str(index * 20)}
        for index in range(count)
    ])


def test_cleaning_and_split():
    raw = raw_rows()
    cleaned = clean_data(pd.concat([raw, raw.iloc[:1]], ignore_index=True))
    assert len(cleaned) == len(raw)
    assert cleaned[TARGET].tolist()[:2] == [0, 1]
    assert cleaned.loc[0, "TotalCharges"] == 0
    assert np.isnan(cleaned.loc[1, "TotalCharges"])
    splits = split_data(cleaned)
    ids = [customer for frame in splits.values() for customer in frame[CUSTOMER_ID]]
    assert len(set(ids)) == len(ids) == len(raw)
    assert [len(frame) for frame in splits.values()] == [28, 6, 6]
    assert all(frame[TARGET].mean() == 0.5 for frame in splits.values())


def test_conflicting_ids_and_invalid_types():
    raw = raw_rows()
    conflict = raw.iloc[:1].assign(Churn="Yes")
    with pytest.raises(ValueError, match="customer ID"):
        clean_data(pd.concat([raw, conflict], ignore_index=True))
    with pytest.raises(ValueError, match="MonthlyCharges"):
        clean_data(raw.assign(MonthlyCharges="broken"))


def test_transform_uses_training_only_and_handles_unknowns():
    raw = raw_rows()
    raw.loc[0, "Partner"] = " "
    data = clean_data(raw)
    assert pd.isna(data.loc[0, "Partner"])
    train, validation = data.iloc[:20], data.iloc[20:].copy()
    validation["Contract"] = "Unseen contract"
    validation["MonthlyCharges"] = 10000
    validation["Partner"] = np.nan
    preprocessor = fit_preprocessor(train)
    imputer = preprocessor.named_transformers_["numeric"].named_steps["simpleimputer"]
    assert imputer.statistics_[2] == train["MonthlyCharges"].median()
    transformed = transform(preprocessor, validation)
    names = preprocessor.get_feature_names_out()
    assert transformed.shape == (20, len(names))
    assert transformed.dtype == np.float32
    assert CUSTOMER_ID not in names and TARGET not in names
    assert transformed[:, names == "Contract_Month-to-month"].sum() == 0
    assert np.isfinite(transformed).all()

def test_reduced_transform_requires_only_the_eight_selected_features():
    data = clean_data(raw_rows())
    preprocessor = fit_preprocessor(data.iloc[:20], REDUCED_FEATURES)
    assert list(preprocessor.feature_names_in_) == REDUCED_FEATURES
    validation = data.iloc[20:]
    np.testing.assert_array_equal(
        transform(preprocessor, validation),
        transform(preprocessor, validation[REDUCED_FEATURES]),
    )
