from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SEED


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Сырой датафрейм со всеми проблемами, которые чинит clean_data"""
    rng = np.random.default_rng(SEED)
    size = 60

    frame = pd.DataFrame(
        {
            "id": range(size),
            "category": "housing/rent/apartment",
            "title": [f"Two BR apartment {i}" for i in range(size)],
            "body": [f"Nice place, only ${1000 + i} per month. Great view." for i in range(size)],
            "amenities": ["Parking,Pool,Gym" if i % 3 else None for i in range(size)],
            "bathrooms": rng.choice([1.0, 1.5, 2.0], size),
            "bedrooms": rng.choice([0.0, 1.0, 2.0, 3.0], size),
            "currency": "USD",
            "fee": "No",
            "has_photo": rng.choice(["Yes", "No", "Thumbnail"], size),
            "pets_allowed": ["Cats,Dogs" if i % 2 else None for i in range(size)],
            "price": rng.uniform(600, 4000, size).round(0),
            "price_display": "$1,250",
            "price_type": "Monthly",
            "square_feet": rng.uniform(400, 2500, size).round(0),
            "address": None,
            "cityname": rng.choice(["Austin", "Denver", "Miami"], size),
            "state": rng.choice(["TX", "CO", "FL"], size),
            "latitude": rng.uniform(25, 48, size).round(4),
            "longitude": rng.uniform(-122, -70, size).round(4),
            "source": "RentDigs.com",
            "time": rng.integers(1_544_000_000, 1_577_000_000, size),
        }
    )

    frame.loc[0, "price"] = np.nan  # пропущенный таргет
    frame.loc[1, "price_type"] = "Weekly"  # другая единица измерения
    frame.loc[1, "price"] = 400.0
    frame.loc[2, "cityname"] = None  # пропущенное гео
    frame.loc[3, "bathrooms"] = np.nan  # пропуск, заполняемый медианой
    frame.loc[4, "price"] = 50_000.0  # выброс по цене
    frame.loc[5, "square_feet"] = 99_999.0  # выброс по площади

    # Полный дубль строки
    return pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
