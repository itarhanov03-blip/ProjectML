"""Очистка сырых данных.

Каждый шаг — отдельная функция, которая логирует, сколько строк потеряла.
Эти цифры идут в отчёт: в CP3 нужна таблица "было -> стало" по каждому шагу.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PRICE_MAX, PRICE_MIN, SQFT_MAX, SQFT_MIN
from src.utils.logging import get_logger

logger = get_logger(__name__)

NUMERIC_COLUMNS = ["bathrooms", "bedrooms", "price", "square_feet", "latitude", "longitude"]

# Сколько недель в месяце: нужно, чтобы привести недельные цены к месячным.
WEEKS_PER_MONTH = 52 / 12


def _log_drop(df_before: pd.DataFrame, df_after: pd.DataFrame, step: str) -> None:
    removed = len(df_before) - len(df_after)
    logger.info("%-28s -%5d строк -> %d", step, removed, len(df_after))


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает полные дубли строк и повторные id.

    Важно сделать это ДО сплита: иначе одно и то же объявление окажется
    одновременно в train и в test, и метрика на тесте будет завышена.
    """
    before = df
    df = df.drop_duplicates()
    _log_drop(before, df, "полные дубли")

    before = df
    df = df.drop_duplicates(subset=["id"], keep="first")
    _log_drop(before, df, "дубли по id")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит числовые колонки к числам, а время — к datetime.

    В сыром файле часть числовых колонок прочиталась как object из-за
    строковых пропусков, поэтому нужен явный errors="coerce".
    """
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # time — unix-таймстамп в секундах.
    df["posted_at"] = pd.to_datetime(df["time"], unit="s", errors="coerce")

    for column in ["category", "fee", "has_photo", "price_type", "currency", "source", "state"]:
        df[column] = df[column].astype("string").str.strip()

    logger.info("Типы приведены")
    return df


def normalize_price_units(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит все цены к месячным.

    В данных 99 488 объявлений с Monthly, 3 с Weekly и 1 с Monthly|Weekly.
    Их всего 4 штуки, но если не заметить, они станут выбросами снизу и
    попадут под отсечение как "ошибки ввода".
    """
    df = df.copy()
    weekly_mask = df["price_type"].fillna("").str.strip() == "Weekly"
    n_weekly = int(weekly_mask.sum())

    df.loc[weekly_mask, "price"] = df.loc[weekly_mask, "price"] * WEEKS_PER_MONTH
    df.loc[weekly_mask, "price_type"] = "Monthly"

    logger.info("Пересчитал недельные цены в месячные: %d строк", n_weekly)
    return df


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Строки без цены бесполезны: восстановить таргет нечем."""
    before = df
    df = df.dropna(subset=["price"])
    _log_drop(before, df, "пропущенный таргет")
    return df


def drop_missing_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает строки без города/штата/координат.

    Таких меньше 0.4%, а гео — одна из сильнейших групп признаков,
    поэтому импутация здесь принесла бы больше шума, чем пользы.
    """
    before = df
    df = df.dropna(subset=["cityname", "state", "latitude", "longitude"])
    _log_drop(before, df, "пропущенное гео")
    return df


def fill_optional(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски там, где само отсутствие значения информативно.

    amenities (16% пропусков) и pets_allowed (61%) — это мультиселект в форме
    подачи объявления. Пустое поле значит "ничего не отметили", а не
    "данные потерялись", поэтому заполняем пустой строкой, а не модой.
    """
    df = df.copy()
    df["amenities"] = df["amenities"].fillna("")
    df["pets_allowed"] = df["pets_allowed"].fillna("")

    # bathrooms/bedrooms: пропусков ~0.1%, медиана не сдвинет распределение.
    for column in ["bathrooms", "bedrooms"]:
        median = df[column].median()
        n_missing = int(df[column].isna().sum())
        df[column] = df[column].fillna(median)
        logger.info("%-28s заполнено медианой %.1f (%d строк)", column, median, n_missing)

    return df


def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Выбрасывает address: 92% пропусков, а гео уже есть в координатах."""
    df = df.drop(columns=["address"], errors="ignore")
    logger.info("Удалил колонку address: %.0f%% пропусков", 92)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Отсекает нереалистичные цены и площади по границам из config.

    Границы выбраны по перцентилям и здравому смыслу, а не по правилу 3 сигм:
    распределение цены сильно скошено вправо, и сигма тут плохой ориентир.
    Разбор — в notebooks/01_eda.ipynb.
    """
    before = df
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    _log_drop(before, df, "выбросы по цене")

    before = df
    df = df[(df["square_feet"] >= SQFT_MIN) & (df["square_feet"] <= SQFT_MAX)]
    _log_drop(before, df, "выбросы по площади")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн очистки."""
    logger.info("Очистка: на входе %d строк", len(df))
    df = (
        df.pipe(drop_duplicates)
        .pipe(cast_types)
        .pipe(normalize_price_units)
        .pipe(drop_missing_target)
        .pipe(drop_missing_geo)
        .pipe(fill_optional)
        .pipe(drop_sparse_columns)
        .pipe(remove_outliers)
        .reset_index(drop=True)
    )
    logger.info("Очистка завершена: %d строк, %d колонок", *df.shape)
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица пропусков по колонкам — для EDA и отчёта."""
    counts = df.isna().sum()
    report = pd.DataFrame(
        {
            "n_missing": counts,
            "pct_missing": (100 * counts / len(df)).round(2),
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(dropna=True),
        }
    )
    return report.sort_values("pct_missing", ascending=False)


def outlier_bounds_iqr(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Границы выбросов по IQR — вспомогательная функция для EDA."""
    q1, q3 = np.nanpercentile(series.dropna(), [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr
