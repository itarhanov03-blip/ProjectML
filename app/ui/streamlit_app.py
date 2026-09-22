"""Интерфейс для оценки аренды.
Запуск:
    streamlit run app/ui/streamlit_app.py
По умолчанию ходит в FastAPI на localhost:8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.data.features import AMENITIES_VOCAB

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="Оценка аренды", page_icon=":)", layout="wide")


@st.cache_data(ttl=300)
def api_available() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data
def load_cities() -> list[str]:
    """Список городов сначала пробуем API, иначе читаем справочник с диска"""
    try:
        response = requests.get(f"{API_URL}/cities", params={"limit": 3000}, timeout=10)
        response.raise_for_status()
        return [c["name"] for c in response.json()["cities"]]
    except requests.RequestException:
        from src.data.reference import load_reference

        counts = load_reference()["city_counts"]
        return sorted(counts, key=counts.get, reverse=True)


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        response.raise_for_status()
        return response.json()["metrics"]
    except requests.RequestException:
        from app.api.predict import get_metrics

        return get_metrics()


def predict(payload: dict) -> dict:
    """Запрос к API, при недоступности — расчёт локально"""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code == 422:
            st.error(f"Данные не приняты: {response.json().get('detail')}")
            st.stop()
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        from app.api.predict import predict_one

        return predict_one(payload)


st.title("Оценка стоимости аренды")
st.caption(
    "Модель обучена на 99 025 объявлениях об аренде жилья в США (UCI Apartment for Rent Classified)"
)

if api_available():
    st.success(f"API доступен: {API_URL}")
else:
    st.warning("API не отвечает, считаю напрямую моделью")

left, right = st.columns([1, 1])

with left:
    st.subheader("Параметры квартиры")

    cities = load_cities()
    city = st.selectbox("Город", cities, index=cities.index("Austin") if "Austin" in cities else 0)

    col1, col2, col3 = st.columns(3)
    square_feet = col1.number_input("Площадь, кв. футы", 100, 6000, 850, step=50)
    bedrooms = col2.number_input("Спальни", 0, 9, 2, help="0 — студия")
    bathrooms = col3.number_input("Санузлы", 1.0, 9.0, 1.0, step=0.5)

    has_photo = st.radio("Фото в объявлении", ["Yes", "Thumbnail", "No"], horizontal=True)

    amenities = st.multiselect("Удобства", AMENITIES_VOCAB, default=["Parking", "Pool"])
    pets = st.multiselect("Питомцы", ["Cats", "Dogs"], default=["Cats"])

    title = st.text_input("Заголовок объявления", "Two BR apartment")
    body = st.text_area(
        "Текст объявления",
        "Renovated apartment close to downtown, quiet street, new kitchen.",
        height=100,
        help="Текст реально влияет на предсказание: он даёт около 15% важности признаков",
    )

    go = st.button("Оценить", type="primary", use_container_width=True)

with right:
    st.subheader("Результат")

    if go:
        payload = {
            "square_feet": float(square_feet),
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "cityname": city,
            "amenities": amenities,
            "pets_allowed": pets,
            "title": title,
            "body": body,
            "has_photo": has_photo,
        }

        with st.spinner("Считаю..."):
            result = predict(payload)

        price = result["predicted_price"]
        low, high = result["price_range"]

        st.metric("Предсказанная аренда", f"${price:,.0f}/мес")
        st.caption(f"Интервал ± MAE модели: ${low:,.0f} — ${high:,.0f}")

        median = result.get("city_median")
        if median:
            diff = price - median
            st.metric(
                f"Медиана по городу {city}",
                f"${median:,.0f}",
                delta=f"{diff:+,.0f} $ к медиане",
                delta_color="off",
            )

        st.info(result["comment"])

        metrics = load_metrics()
        st.markdown("**Качество модели на тесте**")
        st.dataframe(
            pd.DataFrame(
                {
                    "метрика": ["MAE", "RMSE", "MAPE", "R²"],
                    "значение": [
                        f"${metrics['mae']:.1f}",
                        f"${metrics['rmse']:.1f}",
                        f"{metrics['mape']:.1f}%",
                        f"{metrics['r2']:.3f}",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Заполните параметры слева и нажмите «Оценить»")

with st.expander("Как это работает"):
    st.markdown(
        """
        Под капотом XGBoost, обученный на логарифме цены. Признаки:

        - характеристики квартиры: площадь, комнаты, санузлы, производные от них
        - география: координаты города, target encoding города и штата
        - удобства: 27 бинарных флагов
        - текст объявления: TF-IDF со сжатием до 100 компонент

        Из текста заранее вырезаются упоминания цены — иначе модель просто
        читала бы ответ из описания.
        """
    )
