"""Feature engineering.

Исходно в датасете 21 колонка-признак, из них пригодных "как есть" числовых
всего 5. Здесь мы разворачиваем мультиселекты, текст, гео и время
в осмысленные признаки.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import ID_COLUMNS, LEAKY_COLUMNS
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Полный словарь удобств, собранный по сырым данным (27 значений).
AMENITIES_VOCAB = [
    "Parking",
    "Pool",
    "Gym",
    "Patio/Deck",
    "Washer Dryer",
    "Storage",
    "Clubhouse",
    "Dishwasher",
    "AC",
    "Refrigerator",
    "Fireplace",
    "Cable or Satellite",
    "Playground",
    "Internet Access",
    "Wood Floors",
    "Gated",
    "Tennis",
    "TV",
    "Elevator",
    "Basketball",
    "Hot Tub",
    "Garbage Disposal",
    "View",
    "Alarm",
    "Doorman",
    "Luxury",
    "Golf",
]

PETS_VOCAB = ["Cats", "Dogs"]

# Цена в тексте объявления: "$1,250", "1250 / month", "1250$".
PRICE_PATTERN = re.compile(r"\$\s?\d[\d,.]*|\b\d{3,5}\s?(?:\$|usd|per month|/mo|/month)\b", re.I)


def _slug(name: str) -> str:
    """Название удобства -> валидное имя колонки: 'Patio/Deck' -> 'patio_deck'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Разворачивает мультиселект amenities в 27 бинарных флагов + счётчик.

    Само количество удобств — сильный признак: чем длиннее список,
    тем дороже сегмент жилья.
    """
    df = df.copy()
    amenities = df["amenities"].fillna("")

    for name in AMENITIES_VOCAB:
        df[f"am_{_slug(name)}"] = amenities.str.contains(re.escape(name), case=False).astype(int)

    df["amenities_count"] = amenities.apply(lambda s: len([p for p in s.split(",") if p]))
    df["has_amenities"] = (df["amenities_count"] > 0).astype(int)

    logger.info("Добавил %d флагов удобств", len(AMENITIES_VOCAB))
    return df


def add_pets_features(df: pd.DataFrame) -> pd.DataFrame:
    """Флаги разрешённых питомцев.

    Пустое поле трактуем как "не указано" и заводим отдельный флаг:
    отсутствие информации само по себе может коррелировать с ценой.
    """
    df = df.copy()
    pets = df["pets_allowed"].fillna("")

    for name in PETS_VOCAB:
        df[f"pets_{name.lower()}"] = pets.str.contains(name, case=False).astype(int)

    df["pets_not_specified"] = (pets.str.strip() == "").astype(int)
    return df


def strip_prices(text: str) -> str:
    """Вырезает упоминания цены из текста.

    Без этого модель на TF-IDF в CP2 просто прочитает таргет из тела
    объявления: цена там указана почти всегда.
    """
    if not isinstance(text, str):
        return ""
    return PRICE_PATTERN.sub(" ", text)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Базовые статистики по тексту заголовка и описания.

    Векторизация (TF-IDF) — это CP2; здесь только длина и структура,
    чтобы baseline уже мог что-то извлечь из текста.
    """
    df = df.copy()

    df["title_clean"] = df["title"].apply(strip_prices)
    df["body_clean"] = df["body"].apply(strip_prices)

    df["title_len"] = df["title_clean"].str.len()
    df["body_len"] = df["body_clean"].str.len()
    df["body_word_count"] = df["body_clean"].str.split().apply(len)
    df["title_word_count"] = df["title_clean"].str.split().apply(len)

    # Маркеры сегмента, которые часто встречаются в описаниях.
    df["text_luxury"] = (
        df["body_clean"]
        .str.contains(r"luxur|upscale|premium|high[- ]end", case=False, na=False)
        .astype(int)
    )
    df["text_renovated"] = (
        df["body_clean"]
        .str.contains(r"renovat|remodel|updated|newly", case=False, na=False)
        .astype(int)
    )

    logger.info("Добавил текстовые признаки")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Производные от площади и комнат.

    Площадь на спальню разделяет "маленькая трёшка" и "большая трёшка",
    чего не видно по двум колонкам по отдельности.
    """
    df = df.copy()

    df["is_studio"] = (df["bedrooms"] == 0).astype(int)
    rooms = df["bedrooms"].replace(0, 1)  # студию считаем как одну комнату

    df["sqft_per_bedroom"] = df["square_feet"] / rooms
    df["bath_per_bedroom"] = df["bathrooms"] / rooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["sqft_per_room"] = df["square_feet"] / df["total_rooms"].replace(0, 1)
    df["log_square_feet"] = np.log1p(df["square_feet"])

    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Гео-признаки без утечки таргета.

    Кодирование города средней ценой (target encoding) сознательно НЕ делаем
    здесь: его можно считать только внутри фолдов кросс-валидации, иначе
    цена из теста протечёт в обучение. Это задача CP2.
    """
    df = df.copy()

    # Частота города: прокси размера рынка, считается только по признакам.
    city_counts = df["cityname"].value_counts()
    df["city_listing_count"] = df["cityname"].map(city_counts).fillna(0)
    df["state_listing_count"] = df["state"].map(df["state"].value_counts()).fillna(0)

    # Координаты оставляем как есть: деревья умеют резать по ним напрямую.
    df["lat_lon_interaction"] = df["latitude"] * df["longitude"]

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки даты публикации.

    Данные покрывают примерно год (декабрь 2018 — декабрь 2019),
    поэтому год бесполезен, а месяц отражает сезонность рынка аренды.
    """
    df = df.copy()
    df["month"] = df["posted_at"].dt.month
    df["quarter"] = df["posted_at"].dt.quarter
    df["dayofweek"] = df["posted_at"].dt.dayofweek
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает колонки, которые содержат таргет в другом виде.

    price_display — это price строкой ("$2,195"), price_type и currency
    описывают единицы измерения таргета. Если их оставить, любая модель
    покажет идеальный R2 и будет бесполезна в проде.
    """
    to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    logger.info("Удалил колонки с утечкой таргета: %s", ", ".join(to_drop))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн генерации признаков."""
    logger.info("Feature engineering: на входе %d колонок", df.shape[1])
    df = (
        df.pipe(add_amenity_features)
        .pipe(add_pets_features)
        .pipe(add_text_features)
        .pipe(add_ratio_features)
        .pipe(add_geo_features)
        .pipe(add_time_features)
        .pipe(drop_leaky_columns)
    )
    logger.info("Feature engineering: на выходе %d колонок", df.shape[1])
    return df


# Колонки, которые не подаются в модель напрямую: сырой текст, исходные
# мультиселекты и служебные поля. Текст идёт в TF-IDF отдельно в CP2.
NON_FEATURE_COLUMNS = [
    *ID_COLUMNS,
    "title",
    "body",
    "title_clean",
    "body_clean",
    "amenities",
    "pets_allowed",
    "time",
    "posted_at",
    "listing_group",  # служебный id группы объектов, нужен только для сплита
]


def get_feature_columns(df: pd.DataFrame, target: str = "price") -> tuple[list[str], list[str]]:
    """Возвращает (числовые признаки, категориальные признаки)."""
    excluded = set(NON_FEATURE_COLUMNS) | {target}
    candidates = [c for c in df.columns if c not in excluded]

    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical
