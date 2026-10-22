.PHONY: install train prefect-server deploy api docker-build docker-run

install:
	uv sync
	uv run pre-commit install

train:
	uv run python src/modelling/main.py data/abalone.csv

prefect-server:
	uv run prefect server start

deploy:
	uv run python src/modelling/deployment.py

api:
	uv run uvicorn src.web_service.main:app --port 8001

docker-build:
	docker build -t abalone-api -f Dockerfile.app .

docker-run:
	docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
