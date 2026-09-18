"""Разбиение на train/val/test и защита от утечек.

Главная ловушка этого датасета — не случайность сплита, а дубликаты:
одна и та же квартира встречается в нескольких объявлениях с разными id.
При наивном random split почти идентичная строка попадает и в train, и в test,
метрика на тесте оказывается завышенной, а в проде модель работает хуже.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import SEED, TEST_SIZE, VAL_SIZE
from src.utils.logging import get_logger

logger = get_logger(__name__)

# По этим полям считаем, что объявления описывают один и тот же объект.
DEDUP_KEYS = ["latitude", "longitude", "square_feet", "bedrooms", "bathrooms"]


def add_listing_group(df: pd.DataFrame) -> pd.DataFrame:
    """Присваивает объявлениям про один объект общий идентификатор группы.

    Группа = точные координаты + площадь + число комнат и санузлов.
    Совпадение по всем пяти полям при 99 тысячах строк практически наверняка
    означает повторную подачу того же объекта, а не совпадение.
    """
    df = df.copy()
    keys = df[DEDUP_KEYS].round(6).astype(str).agg("|".join, axis=1)
    df["listing_group"] = keys.factorize()[0]

    n_groups = df["listing_group"].nunique()
    logger.info(
        "Групп объектов: %d при %d объявлениях (в среднем %.2f объявления на объект)",
        n_groups,
        len(df),
        len(df) / n_groups,
    )
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
    group_aware: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит данные на train/val/test.

    При group_aware=True все объявления одного объекта целиком уходят
    в одну часть. Это честный сплит; сравнение с наивным (group_aware=False)
    вынесено в notebooks/02 как отдельный эксперимент про даталик.
    """
    if group_aware:
        if "listing_group" not in df.columns:
            df = add_listing_group(df)
        groups = df["listing_group"]

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(df, groups=groups))
        train_val, test = df.iloc[train_val_idx], df.iloc[test_idx]

        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(train_val, groups=train_val["listing_group"]))
        train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]
    else:
        train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
        train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        logger.info(
            "%-5s %6d строк (%4.1f%%) | средняя цена %.0f",
            name,
            len(part),
            100 * len(part) / len(df),
            part["price"].mean(),
        )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def check_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """Считает пересечения между train и test — доказательство честности сплита.

    Возвращает словарь с числом общих групп и общих ключей дедупликации.
    В отчёт идут именно эти цифры: должны быть нули.
    """
    shared_groups = 0
    if "listing_group" in train.columns and "listing_group" in test.columns:
        shared_groups = len(set(train["listing_group"]) & set(test["listing_group"]))

    train_keys = set(train[DEDUP_KEYS].round(6).astype(str).agg("|".join, axis=1))
    test_keys = set(test[DEDUP_KEYS].round(6).astype(str).agg("|".join, axis=1))
    shared_keys = len(train_keys & test_keys)

    result = {"shared_groups": shared_groups, "shared_dedup_keys": shared_keys}
    logger.info("Проверка утечки: %s", result)
    return result


def split_xy(
    df: pd.DataFrame, feature_columns: list[str], target: str = "price"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Разделяет датафрейм на матрицу признаков и вектор таргета."""
    return df[feature_columns], df[target].to_numpy()
