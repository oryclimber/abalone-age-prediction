import pandas as pd

TARGET = "Rings"
CATEGORICAL_FEATURES = ["Sex"]
NUMERICAL_FEATURES = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the abalone dataset from a CSV file."""
    return pd.read_csv(path)


def remove_inconsistent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physically impossible rows (zero/extreme height, shucked weight > whole weight)."""
    is_valid = (
        (df["Height"] > 0)
        & (df["Height"] <= 0.4)
        & (df["Shucked weight"] <= df["Whole weight"])
    )
    return df[is_valid].reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into the features (X) and the target (y)."""
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]
    return X, y
