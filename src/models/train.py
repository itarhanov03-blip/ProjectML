"""Препроцессоры, модели и запуск экспериментов для CP2.

Логика разделения: препроцессор отвечает за то, КАКИЕ признаки видит модель,
модель — за то, КАК она их использует. Это позволяет честно сравнивать
модели между собой на одинаковых данных и отдельно сравнивать наборы
признаков на одинаковой модели.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import SEED, TARGET
from src.data.encoders import (
    HIGH_CARD_CATEGORICAL,
    build_target_encoder,
    build_text_pipeline,
    join_text_columns,
)
from src.data.features import get_feature_columns
from src.models.baseline import LOW_CARD_CATEGORICAL
from src.models.evaluate import compute_metrics, format_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

TEXT_FEATURE = "text_joined"


def add_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет склеенный текст отдельной колонкой.

    ColumnTransformer передаёт в TfidfVectorizer одномерный вход, поэтому
    заголовок и тело объявления объединяются заранее.
    """
    df = df.copy()
    df[TEXT_FEATURE] = join_text_columns(df)
    return df


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Разбивает колонки по способу обработки."""
    numeric, categorical = get_feature_columns(df, target=TARGET)

    return {
        "numeric": [c for c in numeric if c != "listing_group"],
        "low_card": [c for c in categorical if c in LOW_CARD_CATEGORICAL],
        "high_card": [c for c in HIGH_CARD_CATEGORICAL if c in df.columns],
    }


def build_preprocessor(
    df: pd.DataFrame,
    use_target_encoding: bool = True,
    use_text: bool = False,
    scale: bool = True,
    text_components: int = 100,
) -> ColumnTransformer:
    """Собирает препроцессор под нужный набор признаков.

    Все преобразования живут внутри пайплайна, а не применяются к датасету
    заранее: иначе статистики (медианы, средние по городу, словарь TF-IDF)
    посчитаются по всем данным, включая валидацию, и это будет утечка.
    """
    groups = get_column_groups(df)

    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list = [
        ("num", Pipeline(numeric_steps), groups["numeric"]),
        (
            "low_card",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", min_frequency=50, sparse_output=False
                        ),
                    ),
                ]
            ),
            groups["low_card"],
        ),
    ]

    if use_target_encoding and groups["high_card"]:
        transformers.append(("high_card", build_target_encoder(), groups["high_card"]))

    if use_text:
        transformers.append(
            ("text", build_text_pipeline(n_components=text_components), TEXT_FEATURE)
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop", n_jobs=None)

    # Вывод в pandas, а не в numpy. Две причины: имена признаков доходят до
    # модели (важность признаков читается человеком, а не "Column_57"), и
    # LightGBM перестаёт предупреждать о несогласованных именах — его обёртка
    # выставляет feature_names_in_ даже при обучении на массиве.
    return preprocessor.set_output(transform="pandas")


def build_model_zoo(fast: bool = False) -> dict[str, object]:
    """Модели для сравнения: от линейных до бустингов и ансамблей.

    fast=True уменьшает число деревьев — для черновых прогонов,
    когда важна не абсолютная цифра, а порядок величины.
    """
    n_estimators = 100 if fast else 300

    lgbm = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )

    return {
        # --- линейные ---
        "ridge": Ridge(alpha=1.0, random_state=SEED),
        "lasso": Lasso(alpha=0.1, random_state=SEED, max_iter=5000),
        # --- метрические и деревья ---
        "knn_10": KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
        "decision_tree": DecisionTreeRegressor(max_depth=16, min_samples_leaf=5, random_state=SEED),
        # --- бэггинг ---
        "random_forest": rf,
        "extra_trees": ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        # --- бустинги ---
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=0.05, random_state=SEED
        ),
        "xgboost": xgb,
        "lightgbm": lgbm,
        "catboost": CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=0.05,
            depth=8,
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        ),
        # --- ансамбли ---
        "voting_lgbm_xgb_rf": VotingRegressor(
            estimators=[("lgbm", lgbm), ("xgb", xgb), ("rf", rf)], n_jobs=-1
        ),
        "stacking_lgbm_rf_ridge": StackingRegressor(
            estimators=[("lgbm", lgbm), ("rf", rf), ("ridge", Ridge(alpha=1.0))],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        ),
    }


def wrap_log_target(model):
    """Оборачивает модель в обучение на log(price).

    Таргет скошен вправо (асимметрия 9.96 на сырых данных), поэтому модель,
    обученная на логарифме, оптимизирует относительную, а не абсолютную
    ошибку. Предсказания автоматически возвращаются в доллары через expm1,
    так что метрики остаются сравнимыми с остальными экспериментами.
    """
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def build_pipeline(preprocessor, model, log_target: bool = False) -> Pipeline:
    """Собирает препроцессор и модель в один пайплайн."""
    estimator = wrap_log_target(model) if log_target else model
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def run_experiment(
    name: str,
    model,
    preprocessor,
    train: pd.DataFrame,
    val: pd.DataFrame,
    log_target: bool = False,
    features: list[str] | None = None,
) -> dict[str, float]:
    """Обучает пайплайн на train, считает метрики на val, логирует время.

    Время обучения — тоже результат эксперимента: модель, которая выигрывает
    3 доллара MAE ценой получасового обучения, в проде обычно не нужна.
    """
    pipeline = build_pipeline(preprocessor, model, log_target=log_target)

    x_train = train if features is None else train[features]
    x_val = val if features is None else val[features]

    started = time.perf_counter()
    pipeline.fit(x_train, train[TARGET])
    fit_seconds = time.perf_counter() - started

    predictions = pipeline.predict(x_val)
    metrics = compute_metrics(val[TARGET].to_numpy(), predictions)
    metrics["fit_seconds"] = round(fit_seconds, 1)

    logger.info("%-26s %s | обучение %.1f с", name, format_metrics(metrics), fit_seconds)
    return metrics
