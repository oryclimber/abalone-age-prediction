FROM python:3.11-slim
WORKDIR /app_home

COPY pyproject.toml uv.lock /app_home/
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev
ENV PATH="/app_home/.venv/bin:$PATH"

COPY ./src/web_service /app_home/src/web_service
COPY ./bin /app_home/bin

EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
