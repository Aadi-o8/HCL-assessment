import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CATEGORIES, EDA, NUMERIC, TARGET
from src.evaluation import save_figure, write_json
from src.ingestion import strip_strings


def run_eda(raw, train, directory=EDA):
    directory.mkdir(parents=True, exist_ok=True)
    raw_values = strip_strings(raw)
    churn_groups = {}
    for feature in ("Contract", "PaymentMethod", "InternetService"):
        rates = train.groupby(feature)[TARGET].agg(["mean", "size"]).sort_values("mean")
        churn_groups[feature] = rates.to_dict("index")
        ax = rates["mean"].plot.barh(figsize=(8, 4), xlim=(0, 1), title=f"Training churn by {feature}")
        ax.set_xlabel("Churn fraction")
        save_figure(ax.figure, directory, f"churn_by_{feature.lower()}")

    outliers = {}
    for name in NUMERIC:
        values = train[name].dropna()
        lower, upper = values.quantile([0.25, 0.75])
        iqr = upper - lower
        outliers[name] = int(((values < lower - 1.5 * iqr) | (values > upper + 1.5 * iqr)).sum())
    distribution = train[TARGET].value_counts().reindex([0, 1], fill_value=0)
    ax = distribution.rename(index={0: "Stay", 1: "Churn"}).plot.bar(rot=0, title="Training churn distribution")
    ax.set_ylabel("Customers")
    save_figure(ax.figure, directory, "churn_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for feature, ax in zip(("tenure", "MonthlyCharges"), axes):
        sns.boxplot(data=train, x=TARGET, y=feature, ax=ax)
        ax.set_xticks([0, 1], ["Stay", "Churn"])
    save_figure(fig, directory, "charges_tenure_by_churn")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(train[[*NUMERIC, TARGET]].corr(), annot=True, fmt=".2f",
                center=0, cmap="vlag", ax=ax)
    ax.set_title("Training numerical correlations")
    save_figure(fig, directory, "correlations")

    axes = train[list(NUMERIC)].hist(figsize=(9, 6))
    save_figure(axes.flat[0].figure, directory, "numeric_distributions")
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    for feature, ax in zip(CATEGORIES, axes.flat):
        train[feature].value_counts().plot.barh(ax=ax, title=feature)
    axes.flat[-1].set_visible(False)
    save_figure(fig, directory, "category_distributions")

    contract = churn_groups["Contract"]
    highest = max(contract, key=lambda group: contract[group]["mean"])
    statistics = {
        "raw_rows": len(raw), "training_rows": len(train),
        "raw_missing": raw_values.isna().sum().to_dict(),
        "raw_duplicate_rows": int(raw_values.duplicated().sum()),
        "raw_duplicate_customer_ids": int(raw_values["customerID"].dropna().duplicated().sum()),
        "training_churn_fraction": float(train[TARGET].mean()),
        "training_class_counts": {"Stay": int(distribution[0]), "Churn": int(distribution[1])},
        "training_summary": train[list(NUMERIC)].describe().to_dict(),
        "training_outliers_iqr": outliers,
        "training_categories": {name: train[name].value_counts().to_dict() for name in CATEGORIES},
        "training_churn_by_group": churn_groups,
        "training_medians_by_churn": train.groupby(TARGET)[["tenure", "MonthlyCharges"]].median().to_dict(),
        "findings": [
            f"{train[TARGET].mean():.1%} of training customers churned.",
            f"{highest} had the highest observed training churn fraction among contracts "
            f"({contract[highest]['mean']:.1%}, {contract[highest]['size']} customers).",
            "These are descriptive associations in the training sample.",
        ],
    }
    write_json(directory / "summary.json", statistics)
    return statistics
