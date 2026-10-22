from pathlib import Path

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

MODEL_PATH = Path("src/web_service/local_objects/model.pkl")
DATA_DIRPATH = "data/"

TEST_SIZE = 0.2
RANDOM_STATE = 42
