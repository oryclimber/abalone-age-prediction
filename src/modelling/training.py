import numpy as np
import pandas as pd
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from prefect import task
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@task(name="Build pipeline")
def build_pipeline(model) -> Pipeline:
    """Build a pipeline chaining the preprocessing steps and the given regressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


@task(name="Train model")
def train_model(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline) -> Pipeline:
    """Train a linear regression pipeline (preprocessing + model) on the given data."""
    pipeline.fit(X, y)
    return pipeline


@task(name="Evaluate model")
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the regression metrics used to compare the models."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
