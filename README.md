<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
</div>

## 👥 Team (Groupe 5)

| Name | GitHub |
|------|--------|
| Olivia Young | [@oryclimber](https://github.com/oryclimber) |
| Julia Rouzou | [@juliarouzou](https://github.com/juliarouzou) |
| Soro Nangounon Mohamed | [@leroso](https://github.com/leroso) |
| Benedikt Watzinger | [@benedikt-bw](https://github.com/benedikt-bw) |

## 🎯 Project Overview

Determining the age of an abalone (a sea snail) traditionally means cutting its shell, staining it and counting the rings under a microscope. This project predicts the number of rings, and therefore the age (rings + 1.5 years), from easy-to-measure physical features (sex, size and weights).

It contains:
- **EDA and modelling notebooks** (`notebooks/`)
- **A training pipeline** orchestrated with Prefect, with a deployment that retrains the model every week (`src/modelling/`)
- **A prediction API** built with FastAPI and Pydantic, packaged with Docker (`src/web_service/`, `Dockerfile.app`)

## 🚀 Setup

### Prerequisites
- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/get-docker/) (to run the API in a container)
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (only to retrain the model)

### Install

```bash
git clone https://github.com/oryclimber/abalone-age-prediction.git
cd abalone-age-prediction

# Create the virtual environment (.venv), install app + dev dependencies and the pre-commit hooks
make install
```

### Data

Download `abalone.csv` from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and place it at `data/abalone.csv`. It is only needed to retrain the model: a trained model is already committed in `src/web_service/local_objects/model.pkl`.

## ▶️ Running the Training Pipeline (Prefect)

The training code in `src/modelling/` is a [Prefect](https://docs.prefect.io/) flow (`main.py`) made of tasks (loading, preprocessing, training, evaluation, saving). Run every command below **from the repository root**. Each step shows the `make` shortcut and the full command.

### 1. Start the Prefect server and UI

In a dedicated terminal (leave it running):

```bash
make prefect-server   # uv run prefect server start
```

The UI is then available at <http://127.0.0.1:4200>. In another terminal, point Prefect at this server (once):

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

### 2. Run the training flow once

```bash
make train   # uv run python src/modelling/main.py data/abalone.csv
```

This trains the model, prints the test metrics and saves the model to `src/web_service/local_objects/model.pkl`. The flow run and each of its tasks appear in the UI under **Runs**.

### 3. Deploy the flow for regular retraining

```bash
make deploy   # uv run python src/modelling/deployment.py
```

This registers the deployment `Model training Deployment` and keeps a process running that executes it every Sunday at midnight (cron `0 0 * * 0`). In the UI, open **Deployments** to see the schedule, or click **Run → Quick run** to trigger a retraining immediately and follow it in **Runs**.

### Troubleshooting

- `Failed to reach API at http://...:4200/api`: the server is not running (step 1) or `PREFECT_API_URL` points to the wrong address.
- `Found incompatible versions: client ... server ...`: another Prefect install is serving on port 4200. Stop it and start the server with `uv run prefect server start`.
- `fastapi` is pinned below 0.155 in `pyproject.toml` because Prefect 3.8.8 is not compatible with newer FastAPI releases. Run `uv sync` after pulling.

## 🌐 Running the Prediction API

The API (`src/web_service/`) loads `src/web_service/local_objects/model.pkl` and exposes:
- `GET /`: health check
- `POST /predict`: predicts the number of rings and the age of one abalone. Inputs are validated with Pydantic: `sex` must be `M`, `F` or `I` (infant) and all measurements must be positive.

### With Docker (recommended)

The container runs both the API and a Prefect server (`bin/run_services.sh`).

```bash
make docker-build   # docker build -t abalone-api -f Dockerfile.app .
make docker-run     # docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
```

- API docs (try `/predict` from the browser): <http://localhost:8000/docs>
- Prefect UI: <http://localhost:4200>

Stop the container with `Ctrl+C`. Stop any local Prefect server first (`make prefect-server`), as both use port 4200.

### Locally (without Docker)

```bash
make api   # uv run uvicorn src.web_service.main:app --port 8001
```

The API docs are then available at <http://localhost:8001/docs>.

### Example request

With Docker (use port `8001` when running locally):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095, "whole_weight": 0.514, "shucked_weight": 0.2245, "viscera_weight": 0.101, "shell_weight": 0.15}'
```

Response:

```json
{"predicted_rings": 8.90, "predicted_age": 10.40}
```

## 💡 Development

### Managing Dependencies

Use uv to manage dependencies. They are split into two groups in `pyproject.toml`:

- **App dependencies** (`[project] dependencies`): packages the code needs to run (e.g. `pandas`, `scikit-learn`, `pydantic`)
- **Dev dependencies** (`[dependency-groups] dev`): tools only used while developing (e.g. `ruff`, `pre-commit`)

```bash
uv add <package>==<version>        # app dependency
uv add --dev <package>==<version>  # dev dependency
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- Pre-commit hooks (configured in `.pre-commit-config.yaml`) run automatically on every `git commit`:
  - file hygiene checks (trailing whitespace, end-of-file newline, valid TOML/YAML/JSON, no large files)
  - `ruff` linting and `ruff-format` formatting
  - `uv-lock` to keep `uv.lock` in sync with `pyproject.toml`
- If a hook fails or modifies files, the commit is aborted: review the changes, `git add` them and commit again
- Run all hooks manually with `uv run pre-commit run --all-files`
- The same hooks run in CI (`.github/workflows/ci.yaml`) on every push and pull request
