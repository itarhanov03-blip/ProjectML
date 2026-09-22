"""FastAPI-сервис для предсказания стоимости аренды.

Запуск локально:
    uvicorn app.api.main:app --reload
Документация: http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.api.predict import get_metrics, get_model, get_reference, predict_one
from app.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.data.features import AMENITIES_VOCAB
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Модель грузится на старте, а не на первом запросе.

    Иначе первый пользователь ждёт несколько секунд, пока читается
    пайплайн на 44 МБ
    """
    get_model()
    get_reference()
    logger.info("Сервис готов")
    yield


app = FastAPI(
    title="Rental Price API",
    description="Предсказание месячной стоимости аренды жилья по параметрам объявления",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["служебные"])
def health() -> HealthResponse:
    """Проверка, что сервис жив и модель на месте."""
    try:
        model_loaded = get_model() is not None
        cities = len(get_reference()["city_counts"])
    except FileNotFoundError:
        return HealthResponse(status="model missing", model_loaded=False, cities_known=0)

    return HealthResponse(status="ok", model_loaded=model_loaded, cities_known=cities)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["служебные"])
def model_info() -> ModelInfoResponse:
    """Что за модель внутри и с каким качеством."""
    pipeline = get_model()
    estimator = pipeline.named_steps["model"]
    inner = getattr(estimator, "regressor", estimator)

    return ModelInfoResponse(
        model_type=type(inner).__name__,
        metrics={k: round(v, 3) for k, v in get_metrics().items()},
        n_features=len(pipeline.named_steps["prep"].get_feature_names_out()),
        amenities=AMENITIES_VOCAB,
        trained_on="Apartment for Rent Classified, UCI (99 025 объявлений после очистки)",
    )


@app.get("/cities", tags=["справочник"])
def cities(
    q: str = Query("", description="Часть названия города"),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Поиск города по подстроке — для подсказок в интерфейсе"""
    counts = get_reference()["city_counts"]
    query = q.strip().lower()

    matched = [c for c in counts if query in c.lower()] if query else list(counts)
    matched.sort(key=lambda c: counts[c], reverse=True)

    return {
        "total": len(matched),
        "cities": [{"name": c, "listings": counts[c]} for c in matched[:limit]],
    }


@app.post("/predict", response_model=PredictResponse, tags=["предсказание"])
def predict(request: PredictRequest) -> PredictResponse:
    """Предсказание для одного объявления"""
    try:
        return PredictResponse(**predict_one(request.model_dump()))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["предсказание"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Пачка объявлений за один запрос, максимум 100."""
    try:
        predictions = [PredictResponse(**predict_one(item.model_dump())) for item in request.items]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return BatchPredictResponse(predictions=predictions)
