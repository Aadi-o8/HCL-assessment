# Data dictionary

The raw CSV contains one customer per row. Charges use the sample dataset's monetary units.

| Field | Meaning and values |
| --- | --- |
| customerID | Unique raw identifier used to check split overlap. Excluded from model inputs. |
| gender | Female or Male. |
| SeniorCitizen | Senior indicator, 0 or 1. |
| Partner, Dependents | Whether the customer has a partner or dependents, Yes or No. |
| tenure | Non-negative integer months with the company. |
| PhoneService | Yes or No. |
| MultipleLines | Yes, No, or No phone service. |
| InternetService | DSL, Fiber optic, or No. |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Yes, No, or No internet service. |
| Contract | Month-to-month, One year, or Two year. |
| PaperlessBilling | Yes or No. |
| PaymentMethod | Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic). |
| MonthlyCharges | Non-negative monthly charges. |
| TotalCharges | Non-negative cumulative charges. A blank with zero tenure becomes zero. Other blanks are imputed using training statistics. |
| Churn | Target. Raw No becomes 0 and Yes becomes 1. |

The current API requires eight fields: tenure, Contract, MonthlyCharges, TotalCharges, InternetService, PaymentMethod, OnlineSecurity, and TechSupport. The `/models` endpoint lists the active fields. Without internet service, both add-ons must be `No internet service`. With internet service, both must be `Yes` or `No`. Zero tenure requires zero total charges, and positive tenure requires positive total charges. Charges can change over time, so the API accepts cumulative charges independently of the current monthly charge. The raw file also contains the customer ID and target. Split CSVs retain those two columns for auditing and evaluation.
