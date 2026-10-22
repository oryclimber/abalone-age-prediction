# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from preprocessing import load_data, remove_inconsistent_rows, split_features_target
from sklearn.model_selection import train_test_split
from training import evaluate, train_model
from utils import pickle_object

MODEL_PATH = Path("src/web_service/local_objects/model.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = load_data(trainset_path)

    # Preprocess data
    df = remove_inconsistent_rows(df)
    X, y = split_features_target(df)

    # Evaluate the model on a held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    metrics = evaluate(y_test, train_model(X_train, y_train).predict(X_test))
    print(f"Test metrics: {metrics}")

    # Train model on the full dataset
    model = train_model(X, y)

    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    pickle_object(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(Path(args.trainset_path))
