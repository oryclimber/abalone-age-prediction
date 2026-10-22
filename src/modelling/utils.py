import pickle
from pathlib import Path


def pickle_object(obj, path: Path) -> None:
    """Save a Python object (e.g. the trained model) to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
