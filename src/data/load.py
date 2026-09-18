"""Скачивание и чтение сырых данных.

Датасет не хранится в git: он весит ~100 МБ. Вместо этого здесь лежит
воспроизводимая загрузка с UCI — достаточно вызвать `python -m src.data.load`.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    ARCHIVE_NAME,
    CSV_ENCODING,
    CSV_NAME,
    CSV_SEP,
    INNER_7Z_NAME,
    RAW_DIR,
    UCI_ARCHIVE_URL,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# В csv пропуски записаны словом "null", а не пустой строкой.
NA_VALUES = ["null", "NULL", "", "none", "None"]


def download_archive(force: bool = False) -> Path:
    """Скачивает zip-архив с UCI в data/raw. Повторно не качает."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / ARCHIVE_NAME

    if archive_path.exists() and not force:
        logger.info("Архив уже скачан: %s", archive_path)
        return archive_path

    logger.info("Скачиваю %s", UCI_ARCHIVE_URL)
    response = requests.get(UCI_ARCHIVE_URL, timeout=120)
    response.raise_for_status()
    archive_path.write_bytes(response.content)
    logger.info("Сохранил %.1f МБ в %s", len(response.content) / 1e6, archive_path)
    return archive_path


def extract_csv(force: bool = False) -> Path:
    """Распаковывает csv из вложенных архивов: zip -> 7z -> csv."""
    csv_path = RAW_DIR / CSV_NAME
    if csv_path.exists() and not force:
        logger.info("CSV уже распакован: %s", csv_path)
        return csv_path

    archive_path = download_archive(force=force)

    # Первый слой: zip с двумя 7z-архивами (версии на 10K и 100K строк).
    with zipfile.ZipFile(archive_path) as zf:
        zf.extract(INNER_7Z_NAME, RAW_DIR)
    logger.info("Достал %s", INNER_7Z_NAME)

    # Второй слой: 7z с самим csv.
    import py7zr  # импорт здесь, чтобы модуль не требовался для чтения готового csv

    with py7zr.SevenZipFile(RAW_DIR / INNER_7Z_NAME) as z:
        z.extractall(RAW_DIR)
    logger.info("Распаковал %s", csv_path)
    return csv_path


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Возвращает сырой датафрейм: 99 492 строки x 22 колонки."""
    csv_path = extract_csv(force=force_download)
    df = pd.read_csv(
        csv_path,
        sep=CSV_SEP,
        encoding=CSV_ENCODING,
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
    )
    logger.info("Загрузил сырые данные: %d строк, %d колонок", *df.shape)
    return df


if __name__ == "__main__":
    frame = load_raw()
    print(frame.shape)
    print(frame.dtypes)
