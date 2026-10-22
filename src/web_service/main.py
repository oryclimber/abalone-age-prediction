from fastapi import FastAPI

from src.web_service.app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    MODEL_PATH,
)
from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import AbaloneInput, PredictionOut
from src.web_service.utils import load_object

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)
model = load_object(MODEL_PATH)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: AbaloneInput) -> dict:
    rings = run_inference(payload, model)
    return {"predicted_rings": rings, "predicted_age": rings + 1.5}
