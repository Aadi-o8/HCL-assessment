import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORIES, CUSTOMER_ID, FEATURES, NUMERIC, PROCESSED, RAW_DATA, SEED, TARGET,
)


def read_data(path=RAW_DATA):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Download the IBM Telco CSV to {path}. See README.md.")
    return pd.read_csv(path, dtype=str)


def strip_strings(raw):
    return raw.astype("string").apply(lambda column: column.str.strip()).replace("", pd.NA)


def clean_data(raw):
    expected = {CUSTOMER_ID, *FEATURES, TARGET}
    if set(raw.columns) != expected or raw.columns.has_duplicates:
        raise ValueError(f"Unexpected CSV schema. Expected columns: {sorted(expected)}")

    data = strip_strings(raw)
    if data[CUSTOMER_ID].isna().any() or not data[TARGET].isin(["No", "Yes"]).all():
        raise ValueError("Every row needs a customer ID and a No or Yes churn target.")
    for feature, options in CATEGORIES.items():
        if not data[feature].dropna().isin(options).all():
            raise ValueError(f"Invalid category in {feature}.")
        values = data[feature].astype(object)
        data[feature] = values.where(values.notna(), np.nan)
    for feature in NUMERIC:
        values = pd.to_numeric(data[feature], errors="coerce").astype(float)
        invalid = (data[feature].notna() & values.isna()) | (values < 0) | np.isinf(values)
        if feature == "SeniorCitizen":
            invalid |= values.notna() & ~values.isin([0, 1])
        if feature == "tenure":
            invalid |= values.notna() & (values % 1 != 0)
        if invalid.any():
            raise ValueError(f"Invalid numerical value in {feature}.")
        data[feature] = values
    data.loc[data["tenure"].eq(0) & data["TotalCharges"].isna(), "TotalCharges"] = 0.0
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    data = data.drop_duplicates()
    if data[CUSTOMER_ID].duplicated().any():
        raise ValueError("Conflicting rows share a customer ID. Resolve them before splitting.")
    return data.reset_index(drop=True)


def split_data(data):
    train, holdout = train_test_split(
        data, test_size=0.30, stratify=data[TARGET], random_state=SEED,
    )
    validation, test = train_test_split(
        holdout, test_size=0.50, stratify=holdout[TARGET], random_state=SEED,
    )
    return {"train": train, "validation": validation, "test": test}


def save_splits(cleaned, splits, directory=PROCESSED):
    directory.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(directory / "cleaned_data.csv", index=False)
    for name, data in splits.items():
        data.to_csv(directory / f"{name}.csv", index=False)
