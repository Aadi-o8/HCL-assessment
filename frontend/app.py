import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Streamlit runs this file as a script, so make repository imports available.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CATEGORIES, EDA, FIGURES, REPORTS  # noqa: E402

API_URL = os.getenv("CHURNGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")
LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
}


def api_call(path, payload=None):
    try:
        if payload is None:
            response = requests.get(f"{API_URL}{path}", timeout=3)
        else:
            response = requests.post(f"{API_URL}{path}", json=payload, timeout=20)
        if response.status_code == 422 and payload is not None:
            return {"validation_errors": response.json()["detail"]}
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def show_plot(directory, name):
    path = directory / f"{name}.png"
    if path.is_file():
        st.image(str(path), width="stretch")
    else:
        st.info("Run training to generate this figure.")


def overview():
    statistics = read_json(EDA / "summary.json")
    if not statistics:
        st.info("Run training to generate the dataset overview.")
        return
    columns = st.columns(3)
    columns[0].metric("Original customer records", f"{statistics['raw_rows']:,}")
    columns[1].metric("Training records", f"{statistics['training_rows']:,}")
    columns[2].metric("Training churn rate", f"{statistics['training_churn_fraction']:.1%}")
    for finding in statistics["findings"]:
        st.write(finding)
    pair = st.columns(2)
    for column, name in zip(pair, ("churn_distribution", "churn_by_contract")):
        with column:
            show_plot(EDA, name)
    missing = [f"{count:,} in {name}" for name, count in statistics["raw_missing"].items() if count]
    blanks = f"Blank raw values: {', '.join(missing)}." if missing else "The raw data had no missing values."
    duplicate_rows = statistics["raw_duplicate_rows"]
    duplicate_ids = statistics["raw_duplicate_customer_ids"]
    duplicates = (
        f"Data checks found {duplicate_rows:,} duplicate rows and {duplicate_ids:,} repeated customer IDs."
        if duplicate_rows or duplicate_ids else "Data checks found no duplicate rows or customer IDs."
    )
    st.caption(
        f"{duplicates} {blanks} "
        "During cleaning, blank total charges for customers with zero tenure are set to zero."
        if missing else f"{duplicates} {blanks}"
    )
    with st.expander("More dataset plots"):
        for name in ("churn_by_paymentmethod", "churn_by_internetservice",
                     "charges_tenure_by_churn", "correlations",
                     "numeric_distributions", "category_distributions"):
            show_plot(EDA, name)


def customer_input(name, internet):
    label = LABELS.get(name, name)
    if name in CATEGORIES:
        options = CATEGORIES[name]
        if "No internet service" in options:
            if internet == "No":
                return st.selectbox(label, ("No internet service",), disabled=True, key=f"{name}_no_internet")
            options = ("No", "Yes")
        return st.selectbox(label, options, key=name)
    if name == "SeniorCitizen":
        return st.selectbox(label, [0, 1], format_func=lambda n: ("No", "Yes")[n])
    if name == "tenure":
        return st.number_input(label, min_value=0, max_value=1_000_000_000, step=1)
    return st.number_input(label, min_value=0.0, max_value=1_000_000_000.0, step=0.01)


def customer_prediction():
    catalog = api_call("/models")
    if catalog is None:
        st.warning("Start the FastAPI server to enter a prediction.")
        return
    required = catalog.get("required_features")
    if catalog.get("restart_required") or required is None:
        st.warning("Restart FastAPI to load the selected prediction fields.")
        return
    choices = [item["id"] for item in catalog["models"] if item["available"]]
    if not choices:
        st.info("Run training to make models available.")
        return
    selected = catalog["selected_model"]
    default = choices.index(selected) if selected in choices else 0
    # This control is outside the form so Streamlit updates add-on choices immediately.
    internet = st.selectbox("Internet service", CATEGORIES["InternetService"])
    customer = {"InternetService": internet}
    with st.form("prediction"):
        model_id = st.selectbox("Classifier", choices, index=default)
        primary = ("tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod",
                   "OnlineSecurity", "TechSupport")
        columns = st.columns(2)
        for index, name in enumerate(primary):
            with columns[index % 2]:
                customer[name] = customer_input(name, internet)
        additional = [name for name in required if name not in customer]
        if additional:
            with st.expander("Additional customer and service details", expanded=True):
                columns = st.columns(3)
                for index, name in enumerate(additional):
                    with columns[index % 3]:
                        customer[name] = customer_input(name, internet)
        submitted = st.form_submit_button("Estimate churn", type="primary")
    if submitted:
        result = api_call("/predict", {"customer": customer, "model_id": model_id})
        if result is None:
            st.error("Prediction unavailable. Check that the API is running and artifacts are ready.")
        elif "validation_errors" in result:
            for message in dict.fromkeys(error["message"] for error in result["validation_errors"]):
                st.error(message)
        else:
            label = "Churn" if result["churn_class"] else "Stay"
            st.metric(f"{label} prediction", f"{result['churn_score']:.1%}")
            st.caption(f"Estimated churn score from {result['model_id']}. The score has no independent probability calibration.")


def comparison():
    metrics = api_call("/metrics") or read_json(REPORTS / "metrics.json")
    if metrics is None:
        st.info("Run training to generate model comparison results.")
        return
    winner = metrics["selected_model"]
    selected = metrics["validation"][winner]
    st.write(f"**Selected model:** {winner} · {metrics['mode']} run")
    st.caption(
        f"Validation ROC-AUC {selected['roc_auc']:.3f} and F1 {selected['f1']:.3f}. "
        "Selection used ROC-AUC first, F1 second, then alphabetical model ID."
    )
    split = st.radio("Results split", ["validation", "test"], horizontal=True)
    rows = pd.DataFrame(metrics[split]).T.drop(columns=["confusion_matrix"]).astype(float)
    st.dataframe(rows.style.format("{:.3f}"), width="stretch")
    chart = rows.rename_axis("Model").reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(chart, x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1]))
    feature_results = read_json(REPORTS / "feature_comparison.json")
    if feature_results:
        with st.expander("19 vs 8 features on validation"):
            rows = [
                {"Feature set": name, "Model": model, "ROC-AUC": row["roc_auc"], "F1": row["f1"]}
                for name, results in feature_results["validation"].items()
                for model, row in results.items()
            ]
            st.caption(f"Selected feature set: {feature_results['selected_feature_set']}.")
            st.dataframe(pd.DataFrame(rows).style.format({"ROC-AUC": "{:.4f}", "F1": "{:.4f}"}), width="stretch")
    st.subheader("Held-out test plots")
    show_plot(FIGURES, "roc_curves")
    show_plot(FIGURES, "confusion_matrices")
    st.subheader("Neural network learning curves")
    for name in ("basic_mlp", "optimized_mlp"):
        show_plot(FIGURES, name)


def feature_importance():
    show_plot(FIGURES, "random_forest")
    show_plot(FIGURES, "logistic_regression")
    st.write(
        "Each bar represents one input from the prediction form. Random Forest sums impurity "
        "importance across its encoded columns. Logistic Regression shows its largest "
        "absolute coefficient across those columns. Correlated inputs can "
        "share importance. These plots show model associations, with no causal or individual interpretation."
    )


def main():
    st.set_page_config(page_title="ChurnGuard AI", layout="wide")
    st.title("ChurnGuard AI")
    st.caption("Customer churn analysis · IBM Telco sample")
    tabs = st.tabs(["Dataset Overview", "Customer Prediction", "Model Comparison", "Feature Importance"])
    for tab, render in zip(tabs, (overview, customer_prediction, comparison, feature_importance)):
        with tab:
            render()


if __name__ == "__main__":
    main()