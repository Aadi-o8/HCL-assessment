import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    FullPredictionRequest, PredictionResponse, ReducedPredictionRequest, VALIDATION_MESSAGES,
)
from src.config import FEATURE_SETS, MODEL_FILES, REPORTS
from src.prediction import ModelNotReady, available_models, deployment_feature_set, predict, selected_model

app = FastAPI(title="ChurnGuard AI")
# Bind the request schema at startup so OpenAPI lists the deployed fields.
ACTIVE_FEATURE_SET = deployment_feature_set()
PredictionRequest = ReducedPredictionRequest if ACTIVE_FEATURE_SET == "reduced" else FullPredictionRequest


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    # Pydantic errors can contain the submitted input. Return field paths and error types only.
    details = [
        {"loc": item["loc"], "type": item["type"],
         "message": VALIDATION_MESSAGES.get(item["type"], "Check the required fields and allowed values.")}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health")
def health():
    try:
        ready = ACTIVE_FEATURE_SET == deployment_feature_set() and selected_model() in available_models()
    except ModelNotReady:
        ready = False
    return {"status": "ok", "model_ready": ready}


@app.get("/models")
def models():
    try:
        default = selected_model()
    except ModelNotReady:
        default = None
    restart_required = ACTIVE_FEATURE_SET != deployment_feature_set()
    available = [] if restart_required else available_models()
    return {
        "selected_model": default,
        "feature_set": ACTIVE_FEATURE_SET,
        "restart_required": restart_required,
        "required_features": FEATURE_SETS[ACTIVE_FEATURE_SET],
        "models": [{"id": name, "available": name in available} for name in MODEL_FILES],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_customer(request: PredictionRequest):
    if ACTIVE_FEATURE_SET != deployment_feature_set():
        raise HTTPException(status_code=503, detail="Restart the API to load the new prediction schema.")
    try:
        return predict(request.customer.model_dump(), request.model_id)
    except ModelNotReady as error:
        raise HTTPException(status_code=503, detail="Model not ready. Run training first.") from error


@app.get("/metrics")
def metrics():
    try:
        return json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Metrics unavailable. Run training first.") from error