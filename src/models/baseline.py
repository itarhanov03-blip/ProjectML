"""Baseline-модели для CP1

Логика чекпоинта: сначала показать, какое качество даёт самая простая модель
"из коробки" без feature engineering, и только потом сравнивать с ней всё
остальное. Без этой точки отсчёта цифры продвинутых моделей ничего не значат.

Здесь три уровня:
  1. DummyRegressor  — предсказывает медиану, нижняя граница осмысленности;
  2. LinearRegression на 5 исходных числовых колонках — собственно baseline;
  3. LinearRegression / KNN на полном наборе признаков — проверка, что
     feature engineering вообще что-то даёт.

Запуск целиком:  python -m src.models.baseline
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROCESSED_DIR, SEED, TARGET
from src.data.clean import clean_data
from src.data.features import build_features, get_feature_columns
from src.data.load import load_raw
from src.data.split import add_listing_group, check_leakage, split_data
from src.models.evaluate import compute_metrics, format_metrics, metrics_table, save_run
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.warnings import silence_blas_warnings

logger = get_logger(__name__)

# Колонки, которые были в датасете изначально: baseline обязан обходиться ими.
RAW_NUMERIC = ["bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]

# Категориальные признаки с небольшим числом значений. cityname (2 979 значений)
# сознательно не берём: one-hot по нему раздует матрицу, а адекватное
# кодирование города — это target encoding внутри фолдов, то есть CP2.
LOW_CARD_CATEGORICAL = ["category", "fee", "has_photo", "source", "state"]


def build_preprocessor(
    numeric: list[str], categorical: list[str] | None = None
) -> ColumnTransformer:
    """Препроцессор: медианная импутация + шкалирование, one-hot для категорий.

    Импутация и шкалирование обязаны жить внутри пайплайна, а не применяться
    к датасету заранее: иначе статистики посчитаются по всем данным,
    включая тест, и это будет утечка.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_pipeline, numeric)]

    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical))

    return ColumnTransformer(transformers, remainder="drop")


def build_models(numeric: list[str], categorical: list[str] | None = None) -> dict[str, Pipeline]:
    """Набор baseline-моделей на одном и том же препроцессоре."""
    preprocessor = build_preprocessor(numeric, categorical)

    return {
        "dummy_median": Pipeline(
            [("prep", preprocessor), ("model", DummyRegressor(strategy="median"))]
        ),
        "linear_regression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
        "knn_10": Pipeline(
            [("prep", preprocessor), ("model", KNeighborsRegressor(n_neighbors=10, n_jobs=-1))]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    suffix: str = "",
) -> dict[str, dict[str, float]]:
    """Обучает модели на train и считает метрики на val."""
    results: dict[str, dict[str, float]] = {}

    for name, pipeline in models.items():
        pipeline.fit(train[features], train[TARGET])
        predictions = pipeline.predict(val[features])
        metrics = compute_metrics(val[TARGET].to_numpy(), predictions)
        results[f"{name}{suffix}"] = metrics
        logger.info("%-28s %s", f"{name}{suffix}", format_metrics(metrics))

    return results


def prepare_dataset(save: bool = True) -> pd.DataFrame:
    """Полный путь от сырого csv до датафрейма с признаками."""
    df = load_raw()
    df = clean_data(df)
    df = build_features(df)
    df = add_listing_group(df)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / "dataset.parquet"
        df.to_parquet(path, index=False)
        logger.info("Сохранил подготовленные данные: %s", path)

    return df


def main() -> None:
    silence_blas_warnings()
    set_seed(SEED)

    df = prepare_dataset()
    train, val, test = split_data(df)
    check_leakage(train, test)

    numeric, categorical = get_feature_columns(df, target=TARGET)
    categorical = [c for c in categorical if c in LOW_CARD_CATEGORICAL]
    logger.info(
        "Признаков после FE: %d числовых, %d категориальных", len(numeric), len(categorical)
    )

    results: dict[str, dict[str, float]] = {}

    # 1. Baseline: только исходные числовые колонки, никакого FE.
    logger.info("--- Baseline на исходных признаках ---")
    results |= evaluate_models(build_models(RAW_NUMERIC), train, val, RAW_NUMERIC, "_raw")

    # 2. Те же модели, но на всех сгенерированных признаках.
    logger.info("--- Те же модели на признаках после FE ---")
    full_features = numeric + categorical
    results |= evaluate_models(build_models(numeric, categorical), train, val, full_features, "_fe")

    table = metrics_table(results)
    print("\n" + table.to_string())

    save_run(
        "cp1_baseline",
        metrics=results,
        params={
            "seed": SEED,
            "n_raw_features": len(RAW_NUMERIC),
            "n_fe_features": len(full_features),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        notes="CP1: baseline из коробки против тех же моделей после feature engineering. "
        "Валидация на val, тест не трогали.",
    )


if __name__ == "__main__":
    main()
