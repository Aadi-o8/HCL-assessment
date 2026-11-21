from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import SEED
from src.evaluation import classification_metrics


def train_traditional(x_train, y_train, x_validation, y_validation, quick=False):
    # Each tuple is one candidate. The order breaks ties reproducibly.
    logistic_options = [(c, weight) for c in (0.1, 1.0, 10.0)
                        for weight in (None, "balanced")]
    forest_options = [
        (100, None, 1, None), (100, 8, 4, None), (100, 8, 4, "balanced"),
        (150, 12, 2, None), (150, 12, 2, "balanced"), (100, None, 4, "balanced"),
    ]
    if quick:
        logistic_options = logistic_options[:2]
        forest_options = forest_options[:3]
    candidates = {
        "logistic_regression": [
            LogisticRegression(C=c, class_weight=weight, max_iter=1000, random_state=SEED)
            for c, weight in logistic_options
        ],
        "random_forest": [
            RandomForestClassifier(
                n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                class_weight=weight, random_state=SEED,
            )
            for trees, depth, leaf, weight in forest_options
        ],
    }
    winners = {}
    for model_id, models in candidates.items():
        best_key = (-1, -1)
        for model in models:
            model.fit(x_train, y_train)
            scores = model.predict_proba(x_validation)[:, 1]
            metrics = classification_metrics(y_validation, scores)
            key = (metrics["roc_auc"], metrics["f1"])
            if key > best_key:
                winners[model_id] = model
                best_key = key
    return winners