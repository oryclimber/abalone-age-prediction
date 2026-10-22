import numpy as np
import pandas as pd
from preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train a linear regression pipeline (preprocessing + model) on the given data."""
    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)
    return pipeline


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the regression metrics used to compare the models."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
