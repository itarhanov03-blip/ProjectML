from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import LEAKY_COLUMNS, PRICE_MAX, PRICE_MIN, SQFT_MAX, SQFT_MIN
from src.data.clean import WEEKS_PER_MONTH, clean_data
from src.data.features import AMENITIES_VOCAB, build_features, get_feature_columns, strip_prices
from src.data.split import DEDUP_KEYS, add_listing_group, check_leakage, split_data


class TestClean:
    def test_no_missing_target(self, raw_frame):
        assert clean_data(raw_frame)["price"].isna().sum() == 0

    def test_duplicates_removed(self, raw_frame):
        cleaned = clean_data(raw_frame)
        assert cleaned.duplicated().sum() == 0
        assert cleaned["id"].duplicated().sum() == 0

    def test_weekly_price_converted_to_monthly(self, raw_frame):
        cleaned = clean_data(raw_frame)
        converted = cleaned.loc[cleaned["id"] == 1, "price"]
        assert not converted.empty, "строка с недельной ценой не должна отсекаться как выброс"
        assert converted.iloc[0] == 400.0 * WEEKS_PER_MONTH

    def test_outliers_removed(self, raw_frame):
        cleaned = clean_data(raw_frame)
        assert cleaned["price"].between(PRICE_MIN, PRICE_MAX).all()
        assert cleaned["square_feet"].between(SQFT_MIN, SQFT_MAX).all()

    def test_sparse_column_dropped(self, raw_frame):
        assert "address" not in clean_data(raw_frame).columns

    def test_no_missing_values_left(self, raw_frame):
        cleaned = clean_data(raw_frame)
        assert cleaned.isna().sum().sum() == 0


class TestFeatures:
    def test_leaky_columns_are_dropped(self, raw_frame):
        features = build_features(clean_data(raw_frame))
        for column in LEAKY_COLUMNS:
            assert column not in features.columns

    def test_amenity_flags_are_binary(self, raw_frame):
        features = build_features(clean_data(raw_frame))
        flags = [c for c in features.columns if c.startswith("am_")]
        assert len(flags) == len(AMENITIES_VOCAB)
        assert features[flags].isin([0, 1]).all().all()

    def test_amenities_count_matches_flags(self, raw_frame):
        features = build_features(clean_data(raw_frame))
        flags = [c for c in features.columns if c.startswith("am_")]
        assert (features["amenities_count"] == features[flags].sum(axis=1)).all()

    def test_strip_prices_removes_amounts(self):
        assert "$1,250" not in strip_prices("Only $1,250 / month")
        assert "1250$" not in strip_prices("Call now 1250$ today")
        assert "bedroom" in strip_prices("Two bedroom, $900 per month")

    def test_strip_prices_handles_non_string(self):
        assert strip_prices(None) == ""
        assert strip_prices(np.nan) == ""

    def test_service_columns_not_in_features(self, raw_frame):
        """listing_group — служебный id для сплита, модель его видеть не должна."""
        features = add_listing_group(build_features(clean_data(raw_frame)))
        numeric, categorical = get_feature_columns(features)
        assert "listing_group" not in numeric + categorical
        assert "id" not in numeric + categorical


class TestSplit:
    def test_split_proportions(self, raw_frame):
        features = build_features(clean_data(raw_frame))
        train, val, test = split_data(features)
        assert len(train) + len(val) + len(test) == len(features)
        assert len(train) > len(test) > 0
        assert len(val) > 0

    def test_group_split_has_no_leakage(self, raw_frame):
        """Ключевая проверка CP1: объект не может быть и в train, и в test."""
        features = build_features(clean_data(raw_frame))
        train, _, test = split_data(features, group_aware=True)
        assert check_leakage(train, test) == {"shared_groups": 0, "shared_dedup_keys": 0}

    def test_listing_group_is_stable(self, raw_frame):
        """Одинаковые объекты получают одну группу, разные — разные."""
        features = build_features(clean_data(raw_frame))
        grouped = add_listing_group(features)
        keys = grouped[DEDUP_KEYS].round(6).astype(str).agg("|".join, axis=1)
        assert grouped.groupby(keys)["listing_group"].nunique().max() == 1

    def test_naive_split_is_available_for_comparison(self, raw_frame):
        features = build_features(clean_data(raw_frame))
        train, val, test = split_data(features, group_aware=False)
        assert len(train) + len(val) + len(test) == len(features)


class TestMetrics:
    def test_perfect_prediction(self):
        from src.models.evaluate import compute_metrics

        y = np.array([100.0, 200.0, 300.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == 0
        assert metrics["r2"] == 1

    def test_mae_is_mean_absolute_error(self):
        from src.models.evaluate import compute_metrics

        y_true = np.array([100.0, 200.0])
        y_pred = np.array([150.0, 150.0])
        assert compute_metrics(y_true, y_pred)["mae"] == 50.0

    def test_error_analysis_covers_all_rows(self):
        from src.models.evaluate import error_analysis

        rng = np.random.default_rng(0)
        y_true = rng.uniform(500, 5000, 200)
        y_pred = y_true + rng.normal(0, 100, 200)
        table = error_analysis(y_true, y_pred, bins=4)
        assert table["n"].sum() == 200
        assert isinstance(table, pd.DataFrame)
