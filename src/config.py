from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
EDA = REPORTS / "eda"
FIGURES = REPORTS / "figures"
SEED = 42
THRESHOLD = 0.5
TARGET = "Churn"
CUSTOMER_ID = "customerID"

# These are the raw inputs shared by training, the API and the dashboard.
NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
YES_NO = ("No", "Yes")
SERVICE = (*YES_NO, "No internet service")
CATEGORIES = {
    "gender": ("Female", "Male"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": (*YES_NO, "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": SERVICE,
    "OnlineBackup": SERVICE,
    "DeviceProtection": SERVICE,
    "TechSupport": SERVICE,
    "StreamingTV": SERVICE,
    "StreamingMovies": SERVICE,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
FEATURES = [*NUMERIC, *CATEGORIES]
REDUCED_FEATURES = [
    "tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService",
    "PaymentMethod", "OnlineSecurity", "TechSupport",
]
FEATURE_SETS = {"full": FEATURES, "reduced": REDUCED_FEATURES}
MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "basic_mlp": "basic_mlp.keras",
    "optimized_mlp": "optimized_mlp.keras",
}
