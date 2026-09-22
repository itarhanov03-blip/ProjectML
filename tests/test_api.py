"""Тесты API.

Часть тестов требует обученной модели, поэтому если её нет — они
пропускаются, а не падают. Валидацию запросов проверяем всегда:
она от модели не зависит.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.predict import MODEL_PATH
from src.data.reference import REFERENCE_PATH

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and REFERENCE_PATH.exists()),
    reason="нет обученной модели или справочника",
)

VALID_REQUEST = {
    "square_feet": 850,
    "bedrooms": 2,
    "bathrooms": 1,
    "cityname": "Austin",
    "amenities": ["Parking", "Pool"],
    "pets_allowed": ["Cats"],
    "title": "Two BR apartment",
    "body": "Nice apartment close to downtown.",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@needs_model
class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True

    def test_model_info(self, client):
        data = client.get("/model-info").json()
        assert data["model_type"] == "XGBRegressor"
        assert data["n_features"] > 100
        assert len(data["amenities"]) == 27
        # MAE финальной модели на тесте — 160.2 $
        assert 100 < data["metrics"]["mae"] < 250


@needs_model
class TestPredict:
    def test_returns_reasonable_price(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200

        data = response.json()
        # Данные обрезаны границами 200-10000 $, предсказание должно быть внутри
        assert 200 < data["predicted_price"] < 10_000
        low, high = data["price_range"]
        assert low < data["predicted_price"] < high

    def test_bigger_flat_costs_more(self, client):
        """Проверка здравого смысла: студия дешевле трёшки в том же городе."""
        small = {**VALID_REQUEST, "square_feet": 400, "bedrooms": 0}
        big = {**VALID_REQUEST, "square_feet": 1800, "bedrooms": 3, "bathrooms": 2}

        price_small = client.post("/predict", json=small).json()["predicted_price"]
        price_big = client.post("/predict", json=big).json()["predicted_price"]
        assert price_big > price_small

    def test_expensive_city_costs_more(self, client):
        """Сан-Франциско должен быть дороже Уичито при тех же параметрах."""
        cheap = {**VALID_REQUEST, "cityname": "Wichita"}
        pricey = {**VALID_REQUEST, "cityname": "San Francisco"}

        assert (
            client.post("/predict", json=pricey).json()["predicted_price"]
            > client.post("/predict", json=cheap).json()["predicted_price"]
        )

    def test_batch(self, client):
        response = client.post("/predict/batch", json={"items": [VALID_REQUEST] * 3})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 3

    def test_unknown_city_gives_422(self, client):
        response = client.post("/predict", json={**VALID_REQUEST, "cityname": "Атлантида"})
        assert response.status_code == 422


@needs_model
class TestCities:
    def test_search(self, client):
        data = client.get("/cities", params={"q": "san", "limit": 5}).json()
        assert data["total"] > 0
        assert len(data["cities"]) <= 5
        assert all("san" in c["name"].lower() for c in data["cities"])

    def test_sorted_by_listings(self, client):
        cities = client.get("/cities", params={"limit": 10}).json()["cities"]
        counts = [c["listings"] for c in cities]
        assert counts == sorted(counts, reverse=True)


class TestValidation:
    """Валидация работает без модели — pydantic отсекает запрос раньше."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("square_feet", 10),  # меньше нижней границы очистки
            ("square_feet", 99_999),  # больше верхней
            ("bedrooms", -1),
            ("bathrooms", 0),
            ("cityname", ""),
        ],
    )
    def test_rejects_bad_values(self, client, field, value):
        response = client.post("/predict", json={**VALID_REQUEST, field: value})
        assert response.status_code == 422

    def test_rejects_unknown_amenity(self, client):
        response = client.post("/predict", json={**VALID_REQUEST, "amenities": ["Вертолёт"]})
        assert response.status_code == 422

    def test_rejects_unknown_pet(self, client):
        response = client.post("/predict", json={**VALID_REQUEST, "pets_allowed": ["Хомяк"]})
        assert response.status_code == 422

    def test_missing_required_field(self, client):
        payload = {k: v for k, v in VALID_REQUEST.items() if k != "square_feet"}
        assert client.post("/predict", json=payload).status_code == 422
