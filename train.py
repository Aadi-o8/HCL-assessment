import argparse
import random

import joblib
import numpy as np
import pandas as pd

from src.config import CUSTOMER_ID, FEATURE_SETS, MODEL_FILES, MODELS, PROCESSED, REPORTS, SEED, TARGET, THRESHOLD
from src.eda import run_eda
from src.evaluation import (
    classification_metrics, score_models, select_model, write_evaluation, write_json,
)
from src.ingestion import clean_data, read_data, save_splits, split_data
from src.neural_network import train_networks
from src.preprocessing import fit_preprocessor, transform
from src.traditional_ml import train_traditional


def existing_splits(cleaned):
    paths = {name: PROCESSED / f"{name}.csv" for name in ("train", "validation", "test")}
    present = [path.is_file() and path.stat().st_size > 0 for path in paths.values()]
    if not any(present):
        splits = split_data(cleaned)
        save_splits(cleaned, splits)
        return splits
    if not all(present):
        raise ValueError("Restore all three existing split files before training.")
    splits = {name: pd.read_csv(path, dtype={CUSTOMER_ID: str}) for name, path in paths.items()}
    ids = [customer for split in splits.values() for customer in split[CUSTOMER_ID]]
    if len(ids) != len(set(ids)) or set(ids) != set(cleaned[CUSTOMER_ID]):
        raise ValueError("Existing split IDs must be disjoint and cover the cleaned dataset.")
    # Keep the saved membership and row order. Take values from the validated raw data.
    by_id = cleaned.set_index(CUSTOMER_ID)
    return {name: by_id.loc[split[CUSTOMER_ID]].reset_index() for name, split in splits.items()}


def fit_candidates(train, validation, features, quick):
    preprocessor = fit_preprocessor(train, features)
    x_train, x_validation = transform(preprocessor, train), transform(preprocessor, validation)
    data = (x_train, train[TARGET], x_validation, validation[TARGET])
    models = train_traditional(*data, quick=quick)
    networks, histories, tuning_rows = train_networks(*data, quick=quick)
    models.update(networks)
    validation_metrics = {
        name: classification_metrics(validation[TARGET], scores)
        for name, scores in score_models(models, x_validation).items()
    }
    return preprocessor, models, histories, tuning_rows, validation_metrics


def compare_features(full, reduced):
    average_drop = {
        metric: float(np.mean([full[name][metric] - reduced[name][metric] for name in MODEL_FILES]))
        for metric in ("roc_auc", "f1")
    }
    best_auc_drop = max(row["roc_auc"] for row in full.values()) - max(
        row["roc_auc"] for row in reduced.values()
    )
    # Accept up to 1 AUC point and 3 F1 points in the mean, and 1 AUC point for the best model.
    comparable = average_drop["roc_auc"] <= 0.01 and average_drop["f1"] <= 0.03 and best_auc_drop <= 0.01
    return ("reduced" if comparable else "full"), average_drop, best_auc_drop


def main(quick=False, export_h5=False):
    random.seed(SEED)
    np.random.seed(SEED)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    raw = read_data()
    splits = existing_splits(clean_data(raw))
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    print(f"Rows: {len(train)} train, {len(validation)} validation, {len(test)} test", flush=True)
    run_eda(raw, train)

    candidates = {}
    for name, features in FEATURE_SETS.items():
        print(f"Comparing {name}: {len(features)} features", flush=True)
        candidates[name] = fit_candidates(train, validation, features, quick)
    full, reduced = (candidates[name][-1] for name in ("full", "reduced"))
    feature_set, average_drop, best_auc_drop = compare_features(full, reduced)
    comparison = {
        "mode": "quick" if quick else "standard",
        "selected_feature_set": feature_set,
        "mean_drop_full_minus_reduced": average_drop,
        "best_auc_drop_full_minus_reduced": best_auc_drop,
        "validation": {
            features: {name: {key: row[key] for key in ("roc_auc", "f1")}
                       for name, row in results.items()}
            for features, results in (("full", full), ("reduced", reduced))
        },
    }
    write_json(REPORTS / "feature_comparison.json", comparison)
    print(f"Selected {feature_set} features using validation. Mean drops: {average_drop}", flush=True)
    preprocessor, models, histories, tuning_rows, validation_metrics = candidates[feature_set]
    winner = select_model(validation_metrics)
    print(f"Selected {winner} using validation results. Scoring the held-out test set.", flush=True)
    x_test = transform(preprocessor, test)
    test_scores = score_models(models, x_test)
    test_metrics = {
        name: classification_metrics(test[TARGET], scores) for name, scores in test_scores.items()
    }

    write_evaluation(validation_metrics, test_metrics, test[TARGET], test_scores, winner,
                     models, preprocessor, histories, quick)
    write_json(REPORTS / "training_history.json", histories)
    pd.DataFrame(tuning_rows).to_csv(REPORTS / "tuning_results.csv", index=False)

    joblib.dump(preprocessor, MODELS / "preprocessor.joblib")
    for name, model in models.items():
        path = MODELS / MODEL_FILES[name]
        if path.suffix == ".keras":
            model.save(path)
            if export_h5:
                model.save(path.with_suffix(".h5"))
        else:
            joblib.dump(model, path)
    write_json(MODELS / "best_model.json", {
        "model_id": winner, "model_path": f"models/{MODEL_FILES[winner]}",
        "preprocessor_path": "models/preprocessor.joblib",
        "framework": "keras" if MODEL_FILES[winner].endswith(".keras") else "scikit-learn",
        "feature_set": feature_set, "raw_features": FEATURE_SETS[feature_set],
        "input_dimension": x_test.shape[1], "threshold": THRESHOLD,
        "validation_roc_auc": validation_metrics[winner]["roc_auc"],
        "validation_f1": validation_metrics[winner]["f1"],
    })
    print(f"Saved four classifiers and reports. {winner} test ROC-AUC: {test_metrics[winner]['roc_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ChurnGuard AI")
    parser.add_argument("--quick", action="store_true", help="Run a smaller validation search and train for up to three epochs")
    parser.add_argument("--export-h5", action="store_true", help="Also export the two trained ANN models in legacy HDF5 format")
    arguments = parser.parse_args()
    main(arguments.quick, arguments.export_h5)
