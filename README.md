# ChurnGuard AI

ChurnGuard AI compares four classifiers that estimate whether a telecom customer will leave. The dashboard combines exploratory analysis, an individual prediction form and held-out model results for an academic demonstration.

## Run locally

Install [uv](https://docs.astral.sh/uv/). From the repository root, run:

```sh
uv sync
uv run python train.py --quick
uv run python train.py
uv run pytest tests/ -v
```

The project uses Python 3.11. Quick mode reduces the searches and caps each neural fit at three epochs. Standard mode evaluates all six ANN settings and allows up to 50 epochs per fit. Both modes compare the 19- and eight-feature sets using the existing split files when available, and replace the local generated artifacts. Add `--export-h5` to standard training if submission requires legacy copies of the two neural models.

Open two terminals in the same repository:

```sh
uv run uvicorn api.main:app --port 8000
uv run streamlit run frontend/app.py
```

When the virtual environment is already active, use `python -m uvicorn api.main:app --port 8000` and `python -m streamlit run frontend/app.py` in separate terminals. Restart Uvicorn after training if the selected feature set changes so that its request schema updates.

FastAPI serves interactive request examples at `http://127.0.0.1:8000/docs`. Set `CHURNGUARD_API_URL` for Streamlit if the API uses another address. Open the five notebooks with `uv run jupyter notebook`. Run training before opening the evaluation notebook.

## Dataset

The included dataset is the IBM Telco Customer Churn sample with 7,043 rows. If the file is missing, download the IBM CSV or extract `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and save it as `data/raw/telco_customer_churn.csv`. [IBM sample CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) Â· [Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Four classifiers

| Model | What to explain |
| --- | --- |
| Logistic Regression | Scales numeric inputs and uses a linear combination of features to estimate a score. Validation compares regularization and class weights. |
| Random Forest | Averages scores from decision trees. Validation compares tree count, depth, minimum leaf size and class weights. |
| Basic MLP | One hidden layer of 32 ReLU units, followed by a sigmoid output. |
| Optimized MLP | Starts with hidden layers of 64 and 32 units and dropout of 0.3. Six manual settings vary width, dropout and learning rate. |

Both networks use Adam, binary crossentropy, batches of 32 and early stopping. Dropout reduces reliance on particular hidden units. A smaller learning rate takes smaller steps, and wider layers can learn more patterns at higher overfitting risk. The optimized network compares weighted and unweighted training at the initial setting using validation ROC-AUC. The chosen weighting is then used for all six settings. Class weights come from training labels.

## Walk through the pipeline

| File | Responsibility |
| --- | --- |
| `train.py` and `src/config.py` | Run the stages in order and define paths, input fields, model IDs and seed 42. |
| `src/ingestion.py` and `src/eda.py` | Check the raw CSV, clean blanks and duplicates, create stratified 70/15/15 splits, and measure patterns on training rows. |
| `src/preprocessing.py` | Learn imputation, scaling and category levels on training only. Give all classifiers the same float32 inputs and exclude customer ID. |
| `src/traditional_ml.py` and `src/neural_network.py` | Fit candidate settings on training rows and compare them on validation rows. |
| `src/evaluation.py` | Freeze the winner using validation ROC-AUC, then F1, then alphabetical ID. Score all four models on the holdout and write reports. |
| `src/prediction.py`, `api/` and `frontend/app.py` | Validate a raw customer form, load saved artifacts, transform once and return a class and score at threshold 0.5. |

## Results and limitations

The standard run used the same 4,930 training, 1,056 validation, and 1,057 test customers for both feature sets. Each set had its own training-fitted preprocessor. Both sets used identical tuning budgets, seed, and 0.5 threshold.

| Model | 19-feature AUC | 19-feature F1 | 8-feature AUC | 8-feature F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8454 | 0.6182 | 0.8440 | 0.6203 |
| Random Forest | 0.8457 | 0.5823 | 0.8439 | 0.5618 |
| Basic MLP | 0.8483 | 0.5813 | 0.8468 | 0.5818 |
| Optimized MLP | 0.8452 | 0.6121 | 0.8470 | 0.6053 |

The rule accepts the eight-feature set if the mean validation ROC-AUC drop is at most 0.010, the mean F1 drop is at most 0.030, and the best validation ROC-AUC drop is at most 0.010. The respective drops were **0.0007**, **0.0062**, and **0.0013**. Random Forest had the largest F1 drop. The reduced set passed the rule, and all four saved classifiers now use its eight raw inputs.

**Optimized MLP** had the highest validation ROC-AUC among the eight-feature models (0.8470). After both choices were frozen, its test ROC-AUC was 0.8430 and its test F1 was 0.5753. The [feature comparison JSON](reports/feature_comparison.json) contains the exact validation scores. These thresholds are project choices for the demo and the results describe this split.

Read the generated [evaluation report](reports/evaluation_report.md) for validation and test metrics, [tuning table](reports/tuning_results.csv) for configuration results, and [EDA summary](reports/eda/summary.json) for observed data patterns. The dashboard displays the same local results and queries the API for predictions. Training generates the ignored split CSVs, model binaries, deployment metadata and reports.

The figures describe one historical sample. Feature rankings show associations and can shift when features are correlated. The model uses a fixed classification threshold and returns scores without independent probability calibration.

## Team and viva resources

Avinash covers ingestion, EDA and the application. Anish covers preprocessing and works with Atul on traditional ML. Aditya covers the neural networks and tuning. Atul and Aditya share evaluation and model selection.

- [Architecture and inference flow](docs/architecture.md)
- [Data dictionary](docs/data_dictionary.md)
- [Team responsibilities and demo prompts](docs/team_roles.md)
- [Troubleshooting](docs/troubleshooting.md)
