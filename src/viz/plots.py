"""Графики для EDA и отчёта
Каждая функция сохраняет картинку в reports/figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110


def _save(fig: plt.Figure, name: str | None) -> None:
    if name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(FIGURES_DIR) / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")


def plot_target_distribution(df: pd.DataFrame, save_as: str | None = "target_distribution"):
    """Цена и её логарифм рядом.

    Главный вывод для отчёта: распределение скошено вправо, поэтому
    обучение на log(price) — гипотеза, которую стоит проверить в CP2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df["price"], bins=60, ax=axes[0])
    axes[0].set_title("Цена аренды, $/мес")
    axes[0].set_xlabel("price")

    sns.histplot(np.log1p(df["price"]), bins=60, ax=axes[1], color="darkorange")
    axes[1].set_title("log(1 + цена)")
    axes[1].set_xlabel("log1p(price)")

    skew = df["price"].skew()
    fig.suptitle(f"Распределение таргета (асимметрия = {skew:.2f})")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_missing_values(df: pd.DataFrame, save_as: str | None = "missing_values"):
    """Доля пропусков по колонкам."""
    missing = (100 * df.isna().mean()).sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(missing))))
    sns.barplot(x=missing.to_numpy(), y=missing.index, ax=ax, color="steelblue")
    ax.set_title("Пропуски по колонкам, %")
    ax.set_xlabel("% пропусков")

    for i, value in enumerate(missing.to_numpy()):
        ax.text(value + 0.5, i, f"{value:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_price_vs_area(df: pd.DataFrame, sample: int = 5000, save_as: str | None = "price_vs_area"):
    """Цена против площади с разбивкой по числу спален."""
    data = df.sample(min(sample, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x="square_feet",
        y="price",
        hue=data["bedrooms"].clip(upper=4),
        palette="viridis",
        alpha=0.5,
        s=18,
        ax=ax,
    )
    ax.set_title("Цена и площадь (цвет — число спален, 4+ объединены)")
    ax.set_xlabel("square_feet")
    ax.set_ylabel("price, $/мес")
    ax.legend(title="спальни")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_price_by_state(df: pd.DataFrame, top: int = 20, save_as: str | None = "price_by_state"):
    """Медианная цена по штатам: самая сильная группа признаков — гео."""
    stats = (
        df.groupby("state")["price"]
        .agg(["median", "size"])
        .query("size >= 100")
        .sort_values("median", ascending=False)
        .head(top)
    )

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(stats) + 2))
    sns.barplot(x=stats["median"].to_numpy(), y=stats.index, ax=ax, color="teal")
    ax.set_title(f"Медианная цена по штатам (топ-{top}, минимум 100 объявлений)")
    ax.set_xlabel("медианная цена, $/мес")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_geo_scatter(df: pd.DataFrame, sample: int = 20000, save_as: str | None = "geo_scatter"):
    """Объявления на карте координат, цвет — цена.

    Заодно видно выбросы по координатам (Аляска, Гавайи) и плотность рынков.
    """
    data = df.sample(min(sample, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6))
    points = ax.scatter(
        data["longitude"],
        data["latitude"],
        c=data["price"],
        cmap="plasma",
        s=5,
        alpha=0.6,
        vmin=data["price"].quantile(0.05),
        vmax=data["price"].quantile(0.95),
    )
    fig.colorbar(points, ax=ax, label="цена, $/мес")
    ax.set_title("География объявлений")
    ax.set_xlabel("долгота")
    ax.set_ylabel("широта")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_correlations(df: pd.DataFrame, top: int = 20, save_as: str | None = "correlations"):
    """Корреляция числовых признаков с ценой."""
    numeric = df.select_dtypes(include=[np.number])
    correlations = numeric.corr(numeric_only=True)["price"].drop("price")
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    correlations = correlations.head(top)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(correlations) + 2))
    colors = ["indianred" if v < 0 else "seagreen" for v in correlations]
    sns.barplot(
        x=correlations.to_numpy(),
        y=correlations.index,
        ax=ax,
        palette=colors,
        hue=correlations.index,
        legend=False,
    )
    ax.set_title(f"Топ-{top} корреляций с ценой (Пирсон)")
    ax.set_xlabel("корреляция")
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_amenities_effect(df: pd.DataFrame, save_as: str | None = "amenities_effect"):
    """Насколько каждое удобство сдвигает медианную цену."""
    amenity_columns = [c for c in df.columns if c.startswith("am_")]
    overall_median = df["price"].median()

    effects = {
        column.replace("am_", ""): df.loc[df[column] == 1, "price"].median() - overall_median
        for column in amenity_columns
        if df[column].sum() >= 100
    }
    effects = pd.Series(effects).sort_values()

    fig, ax = plt.subplots(figsize=(8, 0.33 * len(effects) + 2))
    colors = ["indianred" if v < 0 else "seagreen" for v in effects]
    sns.barplot(
        x=effects.to_numpy(),
        y=effects.index,
        ax=ax,
        palette=colors,
        hue=effects.index,
        legend=False,
    )
    ax.set_title("Сдвиг медианной цены при наличии удобства, $")
    ax.set_xlabel("разница с общей медианой, $")
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    _save(fig, save_as)
    return fig
