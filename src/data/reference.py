"""Справочник городов для инференса
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

REFERENCE_PATH = MODELS_DIR / "city_reference.json"


def build_reference(df: pd.DataFrame) -> dict:
    """Собирает справочник из обучающих данных."""
    city_counts = df["cityname"].value_counts()
    state_counts = df["state"].value_counts()
    coords = df.groupby("cityname")[["latitude", "longitude"]].median().round(4)

    reference = {
        "city_counts": city_counts.to_dict(),
        "city_medians": df.groupby("cityname")["price"].median().round(0).to_dict(),
        "state_counts": state_counts.to_dict(),
        "city_coords": {
            city: [row["latitude"], row["longitude"]] for city, row in coords.iterrows()
        },
        "city_to_state": df.groupby("cityname")["state"].agg(lambda s: s.mode()[0]).to_dict(),
        "sources": sorted(df["source"].dropna().unique().tolist()),
        "defaults": {
            "category": df["category"].mode()[0],
            "fee": df["fee"].mode()[0],
            "has_photo": df["has_photo"].mode()[0],
            "source": df["source"].mode()[0],
            "median_price": float(df["price"].median()),
        },
    }
    logger.info("Справочник: %d городов, %d штатов", len(city_counts), len(state_counts))
    return reference


def save_reference(reference: dict, path: Path = REFERENCE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    logger.info("Сохранил справочник: %s", path)


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Создать: python -m src.data.reference")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    save_reference(build_reference(df))


if __name__ == "__main__":
    main()
