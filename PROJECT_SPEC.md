# ChurnGuard AI — Implementation specification

Implement a complete, simple, student-friendly customer churn classification project in the existing repository. This is an academic hackathon project; every teammate must be able to understand and defend the code in a viva. Prefer straightforward functions, descriptive names and short explanatory comments. Implement actual working code, not placeholders. Inspect existing files first, preserve useful code, data and Git history, and migrate useful functionality before removing obsolete scaffolding. Do not push changes or perform destructive Git operations.

## Team roles
- Avinash: ingestion, EDA, FastAPI, Streamlit and inference integration.
- Anish: preprocessing and traditional ML with Atul.
- Atul: traditional ML with Anish and evaluation/model selection with Aditya.
- Aditya: neural networks, tuning and evaluation/model selection with Atul.

## Structure
```text
churnguard-ai/
  data/raw/
  data/processed/
  notebooks/01_eda.ipynb
  notebooks/02_preprocessing.ipynb
  notebooks/03_traditional_ml.ipynb
  notebooks/04_neural_network.ipynb
  notebooks/05_model_evaluation.ipynb
  src/__init__.py
  src/config.py
  src/ingestion.py
  src/eda.py
  src/preprocessing.py
  src/traditional_ml.py
  src/neural_network.py
  src/evaluation.py
  src/prediction.py
  models/
  reports/eda/
  reports/figures/
  api/__init__.py
  api/main.py
  api/schemas.py
  frontend/app.py
  tests/test_preprocessing.py
  tests/test_models.py
  tests/test_api.py
  docs/architecture.md
  docs/team_roles.md
  docs/data_dictionary.md
  docs/troubleshooting.md
  train.py
  pyproject.toml
  uv.lock
  .python-version
  .gitignore
  README.md
```
Use Python 3.11, uv, pandas, NumPy, scikit-learn, TensorFlow/Keras, joblib, matplotlib, seaborn, Plotly, FastAPI, Pydantic, Uvicorn, Streamlit, requests, pytest, HTTPX and Jupyter/ipykernel. Use one config.py with repository-relative paths and seed 42. No Optuna, SHAP, Docker, databases, XGBoost, complex registries or other unnecessary frameworks. Run on CPU.

## Data and preprocessing
Use the standard IBM Telco Customer Churn dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn), expected at data/raw/telco_customer_churn.csv. Check local files first. If missing, try a legitimate accessible source or document manual download instructions. Never fabricate training data. The usual 7043-row dataset has columns customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn. Validate schema, types, targets and duplicates. Preserve raw data. Encode Churn No=0, Yes=1. Strip whitespace, convert TotalCharges to numeric; handle zero-tenure blanks appropriately and other missing values using training-fitted imputation. Remove genuine duplicates and prevent duplicate customer IDs crossing splits. Exclude customerID from features. Optionally add number of subscribed services and automatic payment indicator only if consistently computed during inference. Save cleaned data and stratified 70/15/15 train/validation/test CSVs with seed 42. Fit a ColumnTransformer on training only: numerical imputation and StandardScaler, categorical imputation and OneHotEncoder(handle_unknown='ignore'). Save models/preprocessor.joblib. Use the same transformed representation for every classifier; dense float32 for Keras. Do not preprocess twice or leak test information.

## EDA
Implement descriptive stats, missing values, duplicates, class balance, outliers and categorical/numerical distributions. Generate churn distribution, churn by contract/payment/internet service, tenure and monthly charges by churn, and numerical correlation heatmap. Target-aware analysis for model decisions must use training data only. Save plots in reports/eda and report actual findings, not causal claims or invented statistics.

## Exactly four final classifiers
1. logistic_regression: sklearn LogisticRegression, max_iter=1000; small search over C and class weights.
2. random_forest: sklearn RandomForestClassifier, initially 100 trees, small search over depth, leaf size, trees and class weights.
3. basic_mlp: Input -> Dense(32, ReLU, He normal) -> Dense(1, sigmoid).
4. optimized_mlp: initially Input -> Dense(64, ReLU, He normal) -> Dropout(0.3) -> Dense(32, ReLU, He normal) -> Dense(1, sigmoid).
Save classifiers without the preprocessor in models/logistic_regression.joblib, models/random_forest.joblib, models/basic_mlp.keras and models/optimized_mlp.keras. No other final classifiers. Use the validation set for all selection and tuning. Both MLPs use Adam, binary crossentropy, batch size 32, maximum 50 epochs, early stopping patience 5, restore best weights. Basic learning rate .001. Build reusable model functions. Record loss, accuracy, validation curves and best epoch. Compute class weights from training labels only; compare weighted and unweighted optimization using validation ROC-AUC. Preserve original validation/test class proportions. No SMOTE needed.

## Simple ANN tuning
Use six manual configurations (hidden layers, dropout, learning rate):
1. [64,32], 0, .001
2. [64,32], .2, .001
3. [64,32], .3, .001
4. [64,32], .3, .0005
5. [32,16], .3, .001
6. [128,64], .3, .001
Use consistent data, training budgets and chosen class weighting. Clear Keras sessions between trials. Record configurations, validation ROC-AUC, other metrics and best epochs in reports/tuning_results.csv. Save only the best optimized network as a final candidate. Explain dropout, learning rate and capacity effects; do not guarantee improvements. Save final histories in reports/training_history.json.

## Evaluation and selection
Use sklearn metrics across all four models: accuracy, precision, recall, F1, ROC-AUC, confusion matrix. Use continuous scores for ROC-AUC and threshold .5 for class predictions. Handle undefined metrics. Select deployment model using validation ROC-AUC, then validation F1 and deterministic tie-breaking. Freeze selection before evaluating all four models once on the same test set. Do not tune or switch based on test results. Report validation/test results separately. Generate metrics.json, model_comparison.csv, evaluation_report.md under reports; plots under reports/figures: grouped metric comparison, four confusion matrices, combined ROC curves, both ANN loss/accuracy curves, Random Forest feature importance and Logistic Regression coefficients. Use readable feature labels. Explain associations are not causation. No fake individual explanations. Save models/best_model.json with ID, paths, framework, input dimension, threshold, validation ROC-AUC/F1. A traditional model may legitimately win; deliver both ANNs regardless.

## Inference and application
src/prediction.py loads the trusted saved preprocessor and models from a fixed allowlist, reuses loaded artifacts, accepts raw features, transforms once, calls predict_proba or Keras predict, applies .5 threshold. Return model ID, class and estimated churn score (not independently calibrated). No retraining, arbitrary uploaded model files or mock predictions.
FastAPI in api/main.py and api/schemas.py: GET /health, GET /models, POST /predict, GET /metrics. Validate all required raw features, numerical ranges, categories and model IDs with Pydantic. Handle missing artifacts gracefully with model-not-ready errors; API should start before training. Avoid leaking stack traces or customer data.
Streamlit in frontend/app.py: four tabs, Dataset Overview (actual statistics and plots), Customer Prediction (raw feature form, model selector, FastAPI calls, result and estimated score), Model Comparison (actual metrics, ROC, confusion matrices, ANN curves and selected model rationale), Feature Importance (RF importances and LR coefficients with limitations). Use clean, attractive styling with simple Streamlit/Plotly components. Handle unavailable API and missing reports. Batch prediction is optional only after core functionality works.

## Training, notebooks, tests and docs
Root train.py runs ingestion, cleaning and splitting, EDA, preprocessing, traditional models, basic and optimized MLP, validation selection, final test evaluation, reports and artifacts. Support --quick with reduced training budget but identical leakage safeguards. Standard mode runs all six experiments. Print useful stage messages. Set Python/NumPy/TensorFlow seeds as applicable. Generated artifacts must be real, not empty placeholders. Ignore generated binary artifacts and private files in Git; document how to reproduce them.
Complete five executable notebooks with student-friendly explanations and reusable src imports, covering their named stages. The ANN notebook explains forward propagation, backpropagation, ReLU, sigmoid, BCE, Adam, dropout, early stopping, class weights, tuning, curves and serialization. If required for submission, export .h5 copies of the same trained ANN models. Do not fabricate notebook outputs.
Tests: preprocessing target encoding, missing values, dimensions, unknown categories, training-only fit; model shapes/ranges, tiny training smoke test, serialization and metric correctness; API health, invalid inputs, unknown models, missing artifacts and prediction with test artifacts. Synthetic data is only for isolated tests. Tests must not require internet.
README: setup, dataset source, architecture, four models, roles, tuning, commands, actual results and limitations. Docs: architecture Mermaid diagram, data dictionary, team roles and troubleshooting. Explain the pipeline and key files for viva preparation. Keep docs consistent with actual code.

## Verify and report
Run where possible: uv sync; uv run python train.py --quick; uv run python train.py (once if feasible); uv run pytest tests/ -v. Verify saved model loading, prediction, exactly four candidates, actual metrics, leakage prevention, API and UI integration. Startup commands: uv run uvicorn api.main:app --port 8000 and uv run streamlit run frontend/app.py in another terminal. Report actual test/training results and environment blockers honestly. Do not claim unverified success. Prioritize correct, understandable core functionality. Implement incrementally, fix discovered issues and finish with a concise summary, actual metrics if available, selected model rationale, commands, limitations and student-friendly file walkthrough.
