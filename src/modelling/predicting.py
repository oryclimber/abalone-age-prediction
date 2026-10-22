import numpy as np
import pandas as pd
from prefect import task
from sklearn.pipeline import Pipeline


@task(name="Make predictions")
def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict the number of rings for the given abalone measurements."""
    return model.predict(X)
