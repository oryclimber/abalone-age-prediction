# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from predicting import predict
from prefect import flow
from preprocessing import load_data, remove_inconsistent_rows, split_features_target
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from training import build_pipeline, evaluate, train_model
from utils import pickle_object


@flow(name="Train and save model")
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

    eval_pipe = build_pipeline(LinearRegression())
    eval_model = train_model(X_train, y_train, eval_pipe)

    y_pred = predict(eval_model, X_test)
    metrics = evaluate(y_test, y_pred)
    print(f"Test metrics: {metrics}")

    # Build final pipeline
    final_pipe = build_pipeline(LinearRegression())

    # Train final model on the full dataset
    final_model = train_model(X, y, final_pipe)

    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    pickle_object(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(Path(args.trainset_path))
