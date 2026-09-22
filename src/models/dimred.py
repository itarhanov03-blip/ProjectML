"""Снижение размерности и его визуализация.

После feature engineering и кодирования признаков около 160 колонок.
Для бустингов это не проблема — они сами отбирают полезное. А вот метрические
методы (KNN) на таком числе признаков деградируют: в высокой размерности
евклидово расстояние перестаёт различать соседей, и близких точек
практически не остаётся. Это видно в CP1: KNN на 5 признаках дал MAE 301 $,
а на 60 признаках — 323 $.

Здесь проверяется гипотеза: вернёт ли сжатие пространства KNN его силу.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, SEED

PCA_COMPONENTS_GRID = [2, 5, 10, 20, 40, 80]


def fit_pca(x_train: pd.DataFrame, n_components: int | float = 0.95, seed: int = SEED) -> Pipeline:
    """Обучает PCA поверх стандартизации. Дробное n_components — доля дисперсии.

    Масштабирование здесь обязательно, а не косметика. PCA ищет направления
    максимальной дисперсии, поэтому признак, измеренный в долларах, всегда
    перевесит признак, измеренный в нулях и единицах. В нашем препроцессоре
    числовые колонки стандартизованы, а target encoding возвращает цену в
    долларах — без общего масштаба первые компоненты просто повторяют две
    ценовые колонки и "объясняют" 95% дисперсии, ничего не обобщая.
    """
    return Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=seed))]
    ).fit(x_train)


def _as_pca(estimator) -> PCA:
    """Достаёт сам PCA из пайплайна со стандартизацией."""
    return estimator.named_steps["pca"] if isinstance(estimator, Pipeline) else estimator


def explained_variance_table(estimator) -> pd.DataFrame:
    """Таблица накопленной объяснённой дисперсии по компонентам."""
    ratios = _as_pca(estimator).explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1),
            "explained": ratios.round(4),
            "cumulative": ratios.cumsum().round(4),
        }
    )


def plot_explained_variance(estimator, save_as: str | None = "pca_explained_variance"):
    """График накопленной дисперсии: сколько компонент реально нужно."""
    cumulative = _as_pca(estimator).explained_variance_ratio_.cumsum()
    components = np.arange(1, len(cumulative) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(components, cumulative, marker="o", markersize=3)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1, label="95% дисперсии")

    enough = int(np.searchsorted(cumulative, 0.95) + 1)
    ax.axvline(enough, color="gray", linestyle=":", linewidth=1)
    ax.annotate(
        f"{enough} компонент",
        xy=(enough, 0.95),
        xytext=(enough + 3, 0.7),
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )

    ax.set_xlabel("число компонент")
    ax.set_ylabel("накопленная объяснённая дисперсия")
    ax.set_title("PCA: сколько компонент удерживают информацию")
    ax.legend()

    fig.tight_layout()
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{save_as}.png", bbox_inches="tight")
    return fig


def plot_pca_scatter(
    x_transformed: np.ndarray,
    target: np.ndarray,
    sample: int = 8000,
    save_as: str | None = "pca_scatter",
):
    """Первые две компоненты, цвет — цена.

    Если в проекции видны ценовые области, значит сжатие сохранило сигнал,
    а не перемешало всё в однородное облако.
    """
    rng = np.random.default_rng(SEED)
    size = min(sample, len(x_transformed))
    index = rng.choice(len(x_transformed), size=size, replace=False)

    points = x_transformed[index]
    colors = np.asarray(target)[index]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        cmap="plasma",
        s=6,
        alpha=0.6,
        vmin=np.percentile(colors, 5),
        vmax=np.percentile(colors, 95),
    )
    fig.colorbar(scatter, ax=ax, label="цена, $/мес")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Объявления в пространстве двух главных компонент")

    fig.tight_layout()
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{save_as}.png", bbox_inches="tight")
    return fig


def build_pca_pipeline(preprocessor, model, n_components: int, seed: int = SEED) -> Pipeline:
    """Препроцессор -> PCA -> модель.

    PCA стоит внутри пайплайна, чтобы обучаться только на train:
    если сжать весь датасет разом, компоненты вберут информацию о валидации.
    """
    return Pipeline(
        [
            ("prep", preprocessor),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("model", model),
        ]
    )


def plot_dimred_curve(
    results: dict[int, float],
    baseline_mae: float | None = None,
    save_as: str | None = "dimred_knn_curve",
):
    """Качество KNN в зависимости от числа компонент."""
    components = sorted(results)
    scores = [results[c] for c in components]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(components, scores, marker="o", label="KNN на сжатых признаках")

    if baseline_mae is not None:
        ax.axhline(
            baseline_mae,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"KNN на всех признаках ({baseline_mae:.0f} $)",
        )

    best = min(results, key=results.get)
    ax.annotate(
        f"лучшее: {results[best]:.0f} $ при {best} компонентах",
        xy=(best, results[best]),
        xytext=(best, results[best] + (max(scores) - min(scores)) * 0.25),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        ha="center",
    )

    ax.set_xlabel("число компонент PCA")
    ax.set_ylabel("MAE на валидации, $")
    ax.set_title("Снижение размерности и качество KNN")
    ax.legend()

    fig.tight_layout()
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{save_as}.png", bbox_inches="tight")
    return fig
