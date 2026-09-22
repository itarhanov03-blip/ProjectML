FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY tests/ ./tests/
COPY Makefile ruff.toml README.md ./

RUN mkdir -p data/raw data/interim data/processed models reports/figures experiments/runs

CMD ["python", "-m", "src.models.experiments", "--stage", "all"]
