.PHONY: help install data baseline lint format notebooks clean

help:
	@echo "make install    - создать окружение и поставить зависимости"
	@echo "make data       - скачать и подготовить данные"
	@echo "make baseline   - прогнать baseline и сохранить метрики"
	@echo "make lint       - проверить код линтером"
	@echo "make format     - отформатировать код"
	@echo "make notebooks  - выполнить все ноутбуки по порядку"
	@echo "make clean      - удалить кэши и промежуточные файлы"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python -m src.data.load

baseline:
	.venv/bin/python -m src.models.baseline

lint:
	.venv/bin/ruff check src/
	.venv/bin/ruff format --check src/

format:
	.venv/bin/ruff format src/
	.venv/bin/ruff check src/ --fix

notebooks:
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=2400 01_eda.ipynb 02_preprocessing.ipynb 03_baseline.ipynb

clean:
	rm -rf .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
