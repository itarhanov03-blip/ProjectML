"""Кодирование признаков высокой кардинальности и текста.

В CP1 город (2 977 значений) в модель не подавался: one-hot по нему раздувает
матрицу, а кодирование средней ценой в лоб — прямая утечка таргета.
Здесь решаются обе проблемы.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from src.config import SEED

# Признаки высокой кардинальности: 2 977 городов и 51 штат.
HIGH_CARD_CATEGORICAL = ["cityname", "state"]

# Текстовые поля, из которых цена уже вырезана функцией strip_prices (CP1).
TEXT_COLUMNS = ["title_clean", "body_clean"]


def build_target_encoder(cv: int = 5, seed: int = SEED) -> TargetEncoder:
    """Target encoding города и штата с честным cross-fitting.

    Наивный вариант ("средняя цена по городу") — утечка: значение для строки
    считается по выборке, куда входит и сама строка. На train метрика улетает
    вверх, на новых данных модель рассыпается.

    TargetEncoder из sklearn считает кодировку по схеме cross-fitting: датасет
    режется на cv фолдов, и для каждой строки кодировка берётся из статистик,
    посчитанных БЕЗ её фолда. Плюс сглаживание к глобальному среднему, чтобы
    город с тремя объявлениями не получил экстремальную оценку.
    """
    return TargetEncoder(
        target_type="continuous",
        cv=cv,
        smooth="auto",
        random_state=seed,
    )


class TextVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF по тексту объявления со сжатием через SVD.

    Почему сразу со сжатием: голый TF-IDF даёт десятки тысяч разреженных
    колонок. Линейные модели это переварят, а градиентный бустинг на такой
    матрице работает медленно и плохо. SVD (латентно-семантический анализ)
    ужимает текст до n_components плотных компонент, которые бустинг ест
    без проблем.

    Почему отдельным классом, а не Pipeline: TfidfVectorizer не поддерживает
    set_output, и Pipeline с ним ломает вывод ColumnTransformer в pandas.
    Этот класс отдаёт наружу готовые компоненты с именами text_svd_0...N,
    а внутреннюю кухню прячет.

    Важно: на вход подаётся УЖЕ очищенный текст (title_clean / body_clean),
    из которого вырезаны упоминания цены. Иначе модель просто прочитает
    таргет из объявления — см. эксперимент Э1 в experiments.md.
    """

    def __init__(
        self,
        max_features: int = 20_000,
        n_components: int = 100,
        min_df: int = 5,
        seed: int = SEED,
    ):
        self.max_features = max_features
        self.n_components = n_components
        self.min_df = min_df
        self.seed = seed

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        min_df=self.min_df,
                        ngram_range=(1, 2),
                        strip_accents="unicode",
                        lowercase=True,
                        stop_words="english",
                        sublinear_tf=True,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=self.n_components, random_state=self.seed)),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline_ = self._build()
        self.pipeline_.fit(np.asarray(X).ravel())
        return self

    def transform(self, X):
        return self.pipeline_.transform(np.asarray(X).ravel())

    def get_feature_names_out(self, input_features=None):
        return np.array([f"text_svd_{i}" for i in range(self.n_components)])


def build_text_pipeline(
    max_features: int = 20_000,
    n_components: int = 100,
    min_df: int = 5,
    seed: int = SEED,
) -> TextVectorizer:
    """Фабрика TextVectorizer — оставлена ради единообразия с build_target_encoder."""
    return TextVectorizer(
        max_features=max_features, n_components=n_components, min_df=min_df, seed=seed
    )


def join_text_columns(df, columns: list[str] | None = None):
    """Склеивает текстовые колонки в одну строку на объявление.

    TfidfVectorizer принимает одномерную последовательность строк, поэтому
    заголовок и тело объединяем: заголовок обычно содержит тип жилья,
    тело — описание удобств и района.
    """
    columns = columns or TEXT_COLUMNS
    joined = df[columns[0]].fillna("")
    for column in columns[1:]:
        joined = joined + " " + df[column].fillna("")
    return joined
