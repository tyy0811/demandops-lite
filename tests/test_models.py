"""Tests for model implementations."""

from __future__ import annotations

import numpy as np
import pytest

from demandops.features import (
    IDX_DAY_OF_WEEK,
    IDX_HOUR_OF_DAY,
    IDX_LAG_168H,
)
from demandops.models.registry import create_model


def _make_feature_array(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Helper: generate a random feature array with correct column count."""
    return np.column_stack([
        rng.randint(0, 24, n),             # hour_of_day
        rng.randint(0, 7, n),              # day_of_week (0=Mon, 6=Sun)
        rng.randint(0, 2, n),              # is_weekend
        rng.randint(1, 3, n),              # month
        rng.randint(1, 4, n),              # zone_id
        rng.poisson(3, n).astype(float),   # lag_1h
        rng.poisson(3, n).astype(float),   # lag_24h
        rng.poisson(3, n).astype(float),   # lag_168h
        rng.poisson(3, n).astype(float),   # rolling_mean_24h
    ]).astype(float)


class TestSlotMean:

    def test_predictions_non_negative(self) -> None:
        model = create_model("slot_mean")
        rng = np.random.RandomState(42)
        X = _make_feature_array(rng, 100)
        y = rng.poisson(5, 100).astype(float)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert (preds >= 0).all()

    def test_predictions_are_means_of_slots(self) -> None:
        model = create_model("slot_mean")
        # 2 rows for zone=1/hour=0/dow=0 with values 4,6 → mean=5
        X = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # hour=0, dow=0, zone=1
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # hour=0, dow=0, zone=1
            [12, 3, 0, 1, 1, 0, 0, 0, 0], # hour=12, dow=3, zone=1
        ], dtype=float)
        y = np.array([4.0, 6.0, 10.0])
        model.fit(X, y)
        pred = model.predict(np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=float))
        assert abs(pred[0] - 5.0) < 1e-6

    def test_different_zones_produce_different_predictions(self) -> None:
        """Regression: slot mean must key by zone_id, not just hour/dow."""
        model = create_model("slot_mean")
        # Same hour=0/dow=0 but different zones with very different demand
        X = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # zone=1, hour=0, dow=0
            [0, 0, 0, 1, 2, 0, 0, 0, 0],  # zone=2, hour=0, dow=0
        ], dtype=float)
        y = np.array([10.0, 100.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert abs(preds[0] - 10.0) < 1e-6, f"Zone 1 prediction should be 10, got {preds[0]}"
        assert abs(preds[1] - 100.0) < 1e-6, f"Zone 2 prediction should be 100, got {preds[1]}"


class TestSeasonalNaive:

    def test_predictions_equal_lag168(self) -> None:
        model = create_model("seasonal_naive")
        X = np.array([
            [0, 0, 0, 1, 1, 5.0, 10.0, 42.0, 8.0],
            [12, 3, 0, 1, 2, 3.0, 7.0, 99.0, 6.0],
        ], dtype=float)
        y = np.array([0.0, 0.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert abs(preds[0] - 42.0) < 1e-6
        assert abs(preds[1] - 99.0) < 1e-6

    def test_predictions_non_negative(self) -> None:
        model = create_model("seasonal_naive")
        X = np.array([[0, 0, 0, 1, 1, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        y = np.array([0.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()


class TestLightGBM:

    @pytest.fixture
    def trained_lgbm(self) -> tuple:
        rng = np.random.RandomState(42)
        X = _make_feature_array(rng, 200)
        y = rng.poisson(3, 200).astype(float)

        model = create_model(
            "lightgbm", n_estimators=10, num_threads=1,
            random_state=42, verbose=-1,
        )
        model.fit(X[:150], y[:150], eval_set=(X[150:], y[150:]))
        return model, X

    def test_predictions_non_negative(self, trained_lgbm: tuple) -> None:
        model, X = trained_lgbm
        preds = model.predict(X)
        assert (preds >= 0).all(), f"Min prediction: {preds.min()}"

    def test_predictions_near_zero_targets(self) -> None:
        """Even with near-zero targets, predictions are non-negative."""
        rng = np.random.RandomState(42)
        n = 200
        X = _make_feature_array(rng, n)
        # Override lag columns with zeros to push predictions near zero
        X[:, 5:9] = 0.0
        y = rng.uniform(-0.1, 0.5, n).clip(0)

        model = create_model(
            "lightgbm", n_estimators=10, num_threads=1,
            random_state=42, verbose=-1,
        )
        model.fit(X[:150], y[:150], eval_set=(X[150:], y[150:]))
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_predict_raw_returns_unclipped(self, trained_lgbm: tuple) -> None:
        """predict_raw() returns unclipped values (fix #7)."""
        model, X = trained_lgbm
        raw = model.predict_raw(X)
        clipped = model.predict(X)
        # Clipped should be >= 0, raw may have negatives
        assert (clipped >= 0).all()
        # Where raw >= 0, clipped == raw
        mask = raw >= 0
        np.testing.assert_array_almost_equal(clipped[mask], raw[mask])

    def test_deterministic_with_seed(self, trained_lgbm: tuple) -> None:
        model, X = trained_lgbm
        preds1 = model.predict(X[:10])
        preds2 = model.predict(X[:10])
        np.testing.assert_array_equal(preds1, preds2)

    def test_save_and_load_roundtrip(self, trained_lgbm: tuple, tmp_path) -> None:
        """Model survives joblib save/load (fix #4)."""
        model, X = trained_lgbm
        preds_before = model.predict(X[:10])

        path = tmp_path / "model.joblib"
        model.save(path)

        loaded = create_model("lightgbm", num_threads=1, verbose=-1)
        loaded.load(path)
        preds_after = loaded.predict(X[:10])

        np.testing.assert_array_almost_equal(preds_before, preds_after)
