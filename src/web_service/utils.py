import pickle
from pathlib import Path


def load_object(path: Path):
    """Load a Python object (e.g. the trained model) from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
