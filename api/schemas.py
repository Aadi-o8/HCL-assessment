from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from src.config import CATEGORIES

YesNo = Literal["No", "Yes"]
InternetAddOn = Literal["No", "Yes", "No internet service"]
ModelID = Literal["logistic_regression", "random_forest", "basic_mlp", "optimized_mlp"]

VALIDATION_MESSAGES = {
    "internet_service_mismatch": "Internet add-ons must match the selected internet service.",
    "billing_mismatch": "Enter zero total charges for zero months, or positive total charges for positive tenure.",
    "greater_than_equal": "Enter a non-negative value.",
    "less_than_equal": "Enter a value within the supported numeric range.",
    "finite_number": "Enter a finite number.",
}


class ReducedCustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    tenure: int = Field(ge=0, le=1_000_000_000)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    MonthlyCharges: float = Field(ge=0, le=1_000_000_000)
    TotalCharges: float = Field(ge=0, le=1_000_000_000)
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    OnlineSecurity: InternetAddOn
    TechSupport: InternetAddOn

    @model_validator(mode="after")
    def check_consistency(self):
        no_internet = self.InternetService == "No"
        for name, options in CATEGORIES.items():
            if "No internet service" in options and name in type(self).model_fields:
                if (getattr(self, name) == "No internet service") != no_internet:
                    raise PydanticCustomError("internet_service_mismatch", VALIDATION_MESSAGES["internet_service_mismatch"])
        if (self.tenure == 0) != (self.TotalCharges == 0):
            raise PydanticCustomError("billing_mismatch", VALIDATION_MESSAGES["billing_mismatch"])
        return self


class FullCustomerFeatures(ReducedCustomerFeatures):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["No", "Yes", "No phone service"]
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    PaperlessBilling: YesNo


class ReducedPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: ReducedCustomerFeatures
    model_id: ModelID | None = None


class FullPredictionRequest(ReducedPredictionRequest):
    customer: FullCustomerFeatures


class PredictionResponse(BaseModel):
    model_id: ModelID
    churn_class: Literal[0, 1]
    churn_score: float = Field(ge=0, le=1)