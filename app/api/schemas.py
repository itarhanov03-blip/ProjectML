"""Схемы запросов и ответов API"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.data.features import AMENITIES_VOCAB

SQFT_MIN, SQFT_MAX = 100, 6000


class PredictRequest(BaseModel):
    """Параметры квартиры"""

    square_feet: float = Field(..., ge=SQFT_MIN, le=SQFT_MAX, description="Площадь, кв. футы")
    bedrooms: int = Field(..., ge=0, le=9, description="Спальни (0 — студия)")
    bathrooms: float = Field(..., ge=1, le=9, description="Санузлы")
    cityname: str = Field(..., min_length=1, description="Город")
    state: Optional[str] = Field(None, description="Штат, по умолчанию берётся из справочника")

    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    amenities: list[str] = Field(default_factory=list, description="Удобства")
    pets_allowed: list[str] = Field(default_factory=list, description="Cats / Dogs")

    title: str = Field("", max_length=500)
    body: str = Field("", max_length=5000, description="Текст объявления")

    has_photo: str = Field("Yes", description="Yes / No / Thumbnail")

    @field_validator("amenities")
    @classmethod
    def check_amenities(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(AMENITIES_VOCAB)
        if unknown:
            raise ValueError(f"Неизвестные удобства: {sorted(unknown)}")
        return value

    @field_validator("pets_allowed")
    @classmethod
    def check_pets(cls, value: list[str]) -> list[str]:
        unknown = set(value) - {"Cats", "Dogs"}
        if unknown:
            raise ValueError(f"Допустимо только Cats и Dogs, получено: {sorted(unknown)}")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "square_feet": 850,
                "bedrooms": 2,
                "bathrooms": 1,
                "cityname": "Austin",
                "state": "TX",
                "amenities": ["Parking", "Pool", "Gym"],
                "pets_allowed": ["Cats"],
                "title": "Two BR apartment",
                "body": "Renovated apartment close to downtown, quiet street.",
                "has_photo": "Yes",
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_price: float = Field(..., description="Предсказанная аренда, $/мес")
    price_range: list[float] = Field(..., description="Интервал ± MAE модели")
    city_median: Optional[float] = Field(None, description="Медианная цена по городу")
    comment: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cities_known: int


class ModelInfoResponse(BaseModel):
    model_type: str
    metrics: dict
    n_features: int
    amenities: list[str]
    trained_on: str
