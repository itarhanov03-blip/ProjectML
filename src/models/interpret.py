"""Интерпретируемость финальной модели.

Ответ на вопрос "почему модель назвала такую цену" нужен не только для
отчёта. Если топ признаков выглядит бессмысленно, это первый признак
утечки или ошибки в подготовке данных — проверять стоит до, а не после
выкатки модели.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FIGURES_DIR, SEED

# Человекочитаемые названия групп признаков для графиков.
GROUP_PREFIXES = {
    "am_": "удобства",
    "pets_": "питомцы",
    "text_svd_": "текст объявления",
    "high_card__": "город и штат",
    "low_card__": "категориальные",
}


def get_feature_names(pipeline) -> np.ndarray:
    """Имена признаков на выходе препроцессора."""
    return pipeline.named_steps["prep"].get_feature_names_out()


def builtin_importance(pipeline, top: int = 25) -> pd.DataFrame:
    """Встроенная важность признаков модели.

    Для деревьев это суммарное уменьшение ошибки при разбиениях по признаку.
    Метрика быстрая, но смещена в сторону признаков с большим числом
    уникальных значений — поэтому рядом всегда стоит смотреть permutation
    importance.
    """
    model = pipeline.named_steps["model"]
    # Модель может быть обёрнута в обучение на log(price).
    estimator = getattr(model, "regressor_", model)

    importances = estimator.feature_importances_
    names = get_feature_names(pipeline)

    frame = pd.DataFrame({"feature": names, "importance": importances})
    frame["importance"] = frame["importance"] / frame["importance"].sum()
    return frame.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def permutation_based_importance(
    pipeline, x_val: pd.DataFrame, y_val: np.ndarray, n_repeats: int = 5, top: int = 20
) -> pd.DataFrame:
    """Важность через перемешивание значений признака.

    Признак перемешивается случайным образом, и смотрится, насколько
    выросла ошибка. Это честнее встроенной важности: измеряется вклад
    в качество на данных, которых модель не видела, а не в устройство
    самих деревьев.
    """
    result = permutation_importance(
        pipeline,
        x_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=-1,
    )

    frame = pd.DataFrame(
        {
            "feature": x_val.columns,
            "mae_increase": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return frame.sort_values("mae_increase", ascending=False).head(top).reset_index(drop=True)


def group_importance(importance: pd.DataFrame) -> pd.Series:
    """Сворачивает важность признаков в группы.

    Сто компонент текста по отдельности выглядят незначимыми, но вместе
    могут давать больше, чем площадь. Группировка показывает это честно.
    """

    def to_group(name: str) -> str:
        for prefix, label in GROUP_PREFIXES.items():
            if prefix in name:
                return label
        return "числовые характеристики"

    grouped = importance.assign(group=importance["feature"].map(to_group))
    return grouped.groupby("group")["importance"].sum().sort_values(ascending=False)


def plot_importance(
    importance: pd.DataFrame,
    value_column: str = "importance",
    title: str = "Важность признаков",
    save_as: str | None = "feature_importance",
):
    """Горизонтальный график важности."""
    frame = importance.sort_values(value_column)

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(frame) + 2))
    ax.barh(frame["feature"], frame[value_column], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(value_column)

    fig.tight_layout()
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{save_as}.png", bbox_inches="tight")
    return fig


def plot_group_importance(groups: pd.Series, save_as: str | None = "group_importance"):
    """Важность по группам признаков."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(groups.index, groups.to_numpy() * 100, color="teal")
    ax.set_title("Вклад групп признаков, % от суммарной важности")
    ax.set_xlabel("%")

    for i, value in enumerate(groups.to_numpy() * 100):
        ax.text(value + 0.5, i, f"{value:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{save_as}.png", bbox_inches="tight")
    return fig


def plot_prediction_quality(
    y_true: np.ndarray, y_pred: np.ndarray, save_as: str | None = "final_predictions"
):
    """Факт против предсказания и распределение остатков."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_true, y_pred, s=4, alpha=0.2)
    limits = [float(np.min(y_true)), float(np.max(y_true))]
    axes[0].plot(limits, limits, color="red", linewidth=1)
    axes[0].set_xlabel("факт, $")
    axes[0].set_ylabel("предсказание, $")
    axes[0].set_title("Факт против предсказания")

    axes[1].hist(residuals, bins=80, color="steelblue")
    axes[1].axvline(0, color="red", linewidth=1)
    axes[1].set_xlabel("остаток (факт - предсказание), $")
    axes[1].set_title(f"Распределение остатков (медиана {np.median(residuals):.0f} $)")

    fig.tight_layout()
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{save_as}.png", bbox_inches="tight")
    return fig
