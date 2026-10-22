import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict the number of rings for the given abalone measurements."""
    return model.predict(X)
