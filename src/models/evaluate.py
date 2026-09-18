"""Метрики качества и сохранение результатов прогонов.

Основная метрика — MAE в долларах: её можно объяснить одной фразой
("модель ошибается в среднем на N долларов в месяц"), и она устойчива к
выбросам, которых в ценах аренды много.

RMSE считаем как дополнительную: она сильнее штрафует крупные промахи,
и разрыв между MAE и RMSE показывает, есть ли у модели редкие грубые ошибки.
MAPE — относительная ошибка, удобна для сравнения дешёвого и дорогого сегментов.
R2 — только для сопоставимости с внешними работами.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RUNS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

PRIMARY_METRIC = "mae"


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE в процентах. Нулевых цен в данных нет, деление безопасно."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Считает весь набор метрик разом."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Однострочное представление для логов."""
    return (
        f"MAE={metrics['mae']:.1f}$ | RMSE={metrics['rmse']:.1f}$ | "
        f"MAPE={metrics['mape']:.1f}% | R2={metrics['r2']:.3f}"
    )


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Сводная таблица "модель x метрика", отсортированная по основной метрике."""
    table = pd.DataFrame(results).T
    table = table.sort_values(PRIMARY_METRIC)
    return table.round(3)


def save_run(
    name: str,
    metrics: dict[str, dict[str, float]],
    params: dict[str, Any] | None = None,
    notes: str = "",
) -> None:
    """Сохраняет результат прогона в experiments/runs/<name>.json.

    Эти файлы — источник таблицы экспериментов в отчёте, чтобы цифры
    не переписывались руками и не расходились с кодом.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "params": params or {},
        "notes": notes,
    }
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Сохранил результат прогона: %s", path)


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Разбивает ошибку по ценовым квантилям.

    Показывает, на каком сегменте модель ошибается сильнее — без этого
    средний MAE скрывает, что дорогое жильё предсказывается плохо.
    """
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame["price_bin"] = pd.qcut(frame["y_true"], q=bins, duplicates="drop")

    return (
        frame.groupby("price_bin", observed=True)
        .agg(n=("y_true", "size"), mean_price=("y_true", "mean"), mae=("abs_error", "mean"))
        .round(1)
    )
