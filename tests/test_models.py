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
        # 2 rows for hour=0/dow=0 with values 4,6 → mean=5
        X = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # hour=0, dow=0
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # hour=0, dow=0
            [12, 3, 0, 1, 1, 0, 0, 0, 0], # hour=12, dow=3
        ], dtype=float)
        y = np.array([4.0, 6.0, 10.0])
        model.fit(X, y)
        pred = model.predict(np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=float))
        assert abs(pred[0] - 5.0) < 1e-6


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
