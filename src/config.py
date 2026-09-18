"""Пути, константы и параметры проекта.

Всё, что может понадобиться в нескольких модулях, лежит здесь, чтобы не
дублировать магические числа по коду и чтобы эксперимент можно было повторить.
"""

from __future__ import annotations

from pathlib import Path

# --- Воспроизводимость -------------------------------------------------------
SEED = 42

# --- Пути --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"

# --- Источник данных ---------------------------------------------------------
# Apartment for Rent Classified, UCI ML Repository, DOI 10.24432/C5X623,
# лицензия CC BY 4.0. https://archive.ics.uci.edu/dataset/555
UCI_DATASET_ID = 555
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip"

ARCHIVE_NAME = "apartment_for_rent_classified.zip"
INNER_7Z_NAME = "apartments_for_rent_classified_100K.7z"
CSV_NAME = "apartments_for_rent_classified_100K.csv"

# Файл выгружен из СУБД с разделителем ";" в кодировке cp1252 — не UTF-8.
CSV_SEP = ";"
CSV_ENCODING = "cp1252"

# --- Постановка задачи -------------------------------------------------------
TARGET = "price"

# Колонки, которые нельзя подавать в модель: они содержат таргет в другом виде
# либо описывают его единицы измерения. Подробности — в notebooks/02.
LEAKY_COLUMNS = [
    "price_display",  # тот же price, но строкой: "$2,195"
    "price_type",  # единица измерения цены, после нормализации константа
    "currency",  # везде USD, но формально относится к таргету
]

# Технические колонки: идентификатор и сырой адрес (92% пропусков).
ID_COLUMNS = ["id"]

# --- Границы для очистки -----------------------------------------------------
# Обоснование порогов — в notebooks/01_eda.ipynb.
PRICE_MIN = 200  # ниже — явные ошибки ввода и объявления "от"
PRICE_MAX = 10_000  # выше — люкс-сегмент, отдельная задача
SQFT_MIN = 100
SQFT_MAX = 6_000

# --- Сплит -------------------------------------------------------------------
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # доля от оставшейся после отделения test части
