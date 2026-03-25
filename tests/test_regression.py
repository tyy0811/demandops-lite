"""MAE regression gate: blocks merges that silently degrade model quality.

The frozen test fixture and model artifact are committed to the repo.
This test runs in CI on every push. If MAE exceeds the threshold,
something changed that hurt predictions — investigate before merging.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_PATH = Path("artifacts/models/lightgbm.joblib")
FIXTURE_PATH = Path("data/test_fixtures/regression_test.parquet")

# V1 baseline MAE is 2.90. Threshold is 10% above baseline.
# Catches regressions, allows improvements.
MAE_THRESHOLD = 3.20


@pytest.fixture
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    import joblib

    return joblib.load(MODEL_PATH)


@pytest.fixture
def regression_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Regression fixture not found at {FIXTURE_PATH}")
    return pl.read_parquet(FIXTURE_PATH)


class TestRegressionGate:
    def test_mae_below_threshold(self, model, regression_fixture) -> None:
        """MAE must stay below threshold. Prints actual MAE for CI log trend tracking."""
        X = regression_fixture.select(FEATURE_COLUMNS).to_numpy()
        y = regression_fixture[TARGET_COLUMN].to_numpy().astype(float)

        predictions = model.predict(X)
        mae = float(np.mean(np.abs(y - predictions)))

        # Always print MAE so CI logs show the trend over time
        print(f"\n  Regression gate MAE: {mae:.4f} (threshold: {MAE_THRESHOLD})")

        assert mae < MAE_THRESHOLD, (
            f"MAE regression: {mae:.4f} exceeds threshold {MAE_THRESHOLD}. "
            f"V1 baseline was 2.90. Check recent changes to features or data."
        )

    def test_predictions_non_negative(self, model, regression_fixture) -> None:
        """< 5% negative predictions allowed (clipped in serving)."""
        X = regression_fixture.select(FEATURE_COLUMNS).to_numpy()

        predictions = model.predict(X)
        neg_count = int((predictions < 0).sum())

        assert neg_count < len(predictions) * 0.05, (
            f"{neg_count} negative predictions out of {len(predictions)} "
            f"({100 * neg_count / len(predictions):.1f}%)."
        )

    def test_prediction_shape(self, model, regression_fixture) -> None:
        """Output shape matches input shape."""
        X = regression_fixture.select(FEATURE_COLUMNS).to_numpy()
        predictions = model.predict(X)
        assert len(predictions) == len(X)
