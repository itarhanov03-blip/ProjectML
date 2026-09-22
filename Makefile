.PHONY: help install data baseline experiments tune final reference api ui lint format test notebooks docker-build docker-run docker-test docker-app clean

help:
	@echo "make install       - создать окружение и поставить зависимости"
	@echo "make data          - скачать и подготовить данные"
	@echo "make baseline      - baseline из CP1"
	@echo "make experiments   - все эксперименты CP2"
	@echo "make tune          - только подбор гиперпараметров"
	@echo "make final         - обучить финальную модель и оценить на тесте"
	@echo "make reference     - собрать справочник городов для API"
	@echo "make api           - поднять FastAPI на :8000"
	@echo "make ui            - поднять Streamlit на :8501"
	@echo "make lint          - проверить код линтером"
	@echo "make format        - отформатировать код"
	@echo "make test          - прогнать тесты"
	@echo "make notebooks     - выполнить все ноутбуки по порядку"
	@echo "make docker-build  - собрать образ"
	@echo "make docker-run    - прогнать эксперименты в контейнере"
	@echo "make docker-test   - тесты и линтер в контейнере"
	@echo "make docker-app    - поднять api и ui в контейнерах"
	@echo "make clean         - удалить кэши и промежуточные файлы"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python -m src.data.load

baseline:
	.venv/bin/python -m src.models.baseline

experiments:
	.venv/bin/python -m src.models.experiments --stage all

tune:
	.venv/bin/python -m src.models.experiments --stage tuning --trials 25

final:
	.venv/bin/python -m src.models.experiments --stage final

reference:
	.venv/bin/python -m src.data.reference

api:
	.venv/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	.venv/bin/streamlit run app/ui/streamlit_app.py --server.port 8501

lint:
	.venv/bin/ruff check src/ tests/ app/
	.venv/bin/ruff format --check src/ tests/ app/

format:
	.venv/bin/ruff format src/ tests/ app/
	.venv/bin/ruff check src/ tests/ app/ --fix

test:
	.venv/bin/python -m pytest tests/ -v

notebooks:
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 \
		01_eda.ipynb 02_preprocessing.ipynb 03_baseline.ipynb \
		04_experiments.ipynb 05_final_model.ipynb

docker-build:
	docker compose build

docker-run: docker-build
	docker compose run --rm experiments

docker-test: docker-build
	docker compose run --rm test
	docker compose run --rm lint

docker-app: docker-build
	docker compose up api ui

clean:
	rm -rf .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
