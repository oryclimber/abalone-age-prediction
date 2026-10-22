import pandas as pd
from sklearn.pipeline import Pipeline

from src.web_service.lib.models import AbaloneInput

# API field names -> column names the model was trained on
COLUMN_NAMES = {
    "sex": "Sex",
    "length": "Length",
    "diameter": "Diameter",
    "height": "Height",
    "whole_weight": "Whole weight",
    "shucked_weight": "Shucked weight",
    "viscera_weight": "Viscera weight",
    "shell_weight": "Shell weight",
}


def run_inference(payload: AbaloneInput, model: Pipeline) -> float:
    """Predict the number of rings for one abalone."""
    df = pd.DataFrame([payload.model_dump()]).rename(columns=COLUMN_NAMES)
    return float(model.predict(df)[0])
