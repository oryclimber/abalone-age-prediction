import os

from config import DATA_DIRPATH
from main import main
from prefect import serve

if __name__ == "__main__":
    train_model_deployment = main.to_deployment(
        name="Model training Deployment",
        version="0.1.0",
        tags=["training", "model"],
        cron="0 0 * * 0",
        parameters={
            "trainset_path": os.path.join(DATA_DIRPATH, "abalone.csv"),
        },
    )

    serve(train_model_deployment)
