import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from src.config import FIGURES, MODEL_FILES, REPORTS, THRESHOLD

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def classification_metrics(labels, scores):
    predictions = (np.asarray(scores) >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def churn_scores(model, model_id, features):
    if MODEL_FILES[model_id].endswith(".keras"):
        return model.predict(features, verbose=0).ravel()
    return model.predict_proba(features)[:, 1]


def score_models(models, features):
    return {name: churn_scores(model, name, features) for name, model in models.items()}


def select_model(validation):
    # Alphabetical model ID is the final tie-break.
    return min(validation, key=lambda name: (
        -validation[name]["roc_auc"], -validation[name]["f1"], name,
    ))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_evaluation(validation, test, test_labels, test_scores, winner,
                     models, preprocessor, histories, quick=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "metrics.json", {
        "mode": "quick" if quick else "standard", "selected_model": winner,
        "threshold": THRESHOLD, "validation": validation, "test": test,
    })
    rows = [
        {"split": split, "model": name, **{metric: values[metric] for metric in METRICS}}
        for split, results in (("validation", validation), ("test", test))
        for name, values in results.items()
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    lines = [
        "# Model evaluation", "",
        f"Run mode: {'quick' if quick else 'standard'}.",
        f"Selected model: `{winner}`.",
        "Selection used validation ROC-AUC, then validation F1, then alphabetical model ID.",
        "Selection was frozen before test scoring. The classification threshold was 0.5.",
    ]
    for split in ("validation", "test"):
        lines.extend(["", f"## {split.title()}", "",
                      "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for row in comparison[comparison["split"] == split].to_dict("records"):
            values = " | ".join(f"{row[metric]:.4f}" for metric in METRICS)
            lines.append(f"| {row['model']} | {values} |")
    lines.extend([
        "", "Feature rankings describe model associations. Correlated features can share",
        "or shift importance. Neither plot establishes causation or explains a specific prediction.",
        "Scores are estimates without independent probability calibration.", "",
    ])
    (REPORTS / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    ax = comparison[comparison["split"] == "test"].set_index("model").drop(columns="split").T.plot.bar(
        figsize=(10, 5), rot=0, ylim=(0, 1), title="Test metric comparison",
    )
    save_figure(ax.figure, FIGURES, "metric_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for (name, values), ax in zip(test.items(), axes.flat):
        ConfusionMatrixDisplay(np.array(values["confusion_matrix"]), display_labels=["Stay", "Churn"]).plot(
            ax=ax, colorbar=False,
        )
        ax.set_title(name.replace("_", " "))
    save_figure(fig, FIGURES, "confusion_matrices")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores in test_scores.items():
        RocCurveDisplay.from_predictions(test_labels, scores, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Test ROC curves")
    save_figure(fig, FIGURES, "roc_curves")

    for name, history in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for metric, ax in zip(("loss", "accuracy"), axes):
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], label="Training")
            ax.plot(epochs, history[f"val_{metric}"], label="Validation")
            ax.axvline(history["best_epoch"], linestyle=":", color="gray", label="Restored epoch")
            ax.set(xlabel="Epoch", ylabel=metric.title(), title=metric.title())
            ax.legend()
        fig.suptitle(name.replace("_", " "))
        save_figure(fig, FIGURES, name)

    plot_feature_importance(models, preprocessor)


def plot_feature_importance(models, preprocessor):
    """Show one bar per original input, including all of its encoded columns."""
    encoded = preprocessor.get_feature_names_out()
    groups = {
        name: [index for index, column in enumerate(encoded)
               if column == name or column.startswith(f"{name}_")]
        for name in preprocessor.feature_names_in_
    }
    for model_id, values, summarize, title in [
        ("random_forest", models["random_forest"].feature_importances_, np.sum,
         "Random Forest impurity importance"),
        ("logistic_regression", models["logistic_regression"].coef_[0],
         lambda group: np.max(np.abs(group)), "Logistic Regression coefficient magnitude"),
    ]:
        scores = {name: summarize(values[indices]) for name, indices in groups.items()}
        ordered = sorted(scores, key=scores.get)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(ordered))))
        ax.barh(ordered, [scores[name] for name in ordered])
        ax.set_title(title)
        save_figure(fig, FIGURES, model_id)
