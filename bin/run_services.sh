#!/bin/bash

# Start the Prefect server in the background, then the API in the foreground

prefect server start --host 0.0.0.0 --port 4201 &
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
