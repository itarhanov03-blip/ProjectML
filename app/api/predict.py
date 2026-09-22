"""Загрузка модели и предсказание по одному объявлению"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache

import joblib
import pandas as pd

from src.config import MODELS_DIR, RUNS_DIR
from src.data.features import build_features
from src.data.reference import load_reference
from src.models.train import add_text_column
from src.utils.logging import get_logger

logger = get_logger(__name__)

get_logger("src.data.features", level=logging.WARNING)

MODEL_PATH = MODELS_DIR / "final_model.pkl"
FALLBACK_MAE = 160.2


@lru_cache(maxsize=1)
def get_model():
    """Модель грузится один раз и живёт в памяти процесса."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Нет {MODEL_PATH}. Обучить: python -m src.models.experiments --stage final"
        )
    logger.info("Загружаю модель: %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_reference() -> dict:
    return load_reference()


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    """Метрики финальной модели из сохранённого прогона."""
    path = RUNS_DIR / "cp2_final.json"
    if not path.exists():
        return {"mae": FALLBACK_MAE}

    run = json.loads(path.read_text(encoding="utf-8"))
    return next(iter(run["metrics"].values()))


def to_raw_row(payload: dict) -> pd.DataFrame:
    """Превращает запрос в строку такого же вида, как в исходном датасете.

    build_features умеет работать только с сырыми колонками, поэтому
    сначала собираем их, а потом прогоняем тот же код, что и при обучении
    """
    reference = get_reference()
    city = payload["cityname"]

    state = payload.get("state") or reference["city_to_state"].get(city)
    coords = reference["city_coords"].get(city)

    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if latitude is None or longitude is None:
        if coords is None:
            raise ValueError(
                f"Город '{city}' не встречался в обучающих данных — "
                "укажите latitude и longitude вручную"
            )
        latitude, longitude = coords

    defaults = reference["defaults"]

    return pd.DataFrame(
        [
            {
                "id": 0,
                "category": defaults["category"],
                "title": payload.get("title", ""),
                "body": payload.get("body", ""),
                "amenities": ",".join(payload.get("amenities", [])),
                "bathrooms": float(payload["bathrooms"]),
                "bedrooms": float(payload["bedrooms"]),
                "fee": defaults["fee"],
                "has_photo": payload.get("has_photo", defaults["has_photo"]),
                "pets_allowed": ",".join(payload.get("pets_allowed", [])),
                "square_feet": float(payload["square_feet"]),
                "cityname": city,
                "state": state,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "source": defaults["source"],
                "posted_at": pd.Timestamp(datetime.now()),
            }
        ]
    )


def build_model_input(payload: dict) -> pd.DataFrame:
    """Сырая строка -> признаки, которые ждёт модель."""
    frame = build_features(to_raw_row(payload))
    frame = add_text_column(frame)
    reference = get_reference()
    frame["city_listing_count"] = reference["city_counts"].get(payload["cityname"], 0)
    frame["state_listing_count"] = reference["state_counts"].get(frame["state"].iloc[0], 0)

    return frame


def predict_one(payload: dict) -> dict:
    """Предсказание с интервалом и справочной ценой по городу."""
    model_input = build_model_input(payload)
    price = float(get_model().predict(model_input)[0])

    mae = float(get_metrics().get("mae", FALLBACK_MAE))
    city_counts = get_reference()["city_counts"]
    known_city = payload["cityname"] in city_counts

    comment = (
        f"Город известен модели ({city_counts[payload['cityname']]} объявлений в обучении)"
        if known_city
        else "Города не было в обучающих данных, точность ниже обычной"
    )

    city_median = get_reference()["city_medians"].get(payload["cityname"])

    return {
        "predicted_price": round(price, 2),
        "price_range": [round(max(price - mae, 0), 2), round(price + mae, 2)],
        "city_median": city_median,
        "comment": comment,
    }
