"""Characterization test: captures prepare() output before refactoring.

Run BEFORE the DatasetAdapter extraction to establish the baseline.
Run AFTER to verify the refactor didn't change behavior.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from demandops.data.prepare import engineer_features


class TestEngineerFeaturesCharacterization:
    """Verify engineer_features() produces identical output before and after refactor."""

    @pytest.fixture
    def small_dense_grid(self) -> pl.DataFrame:
        """Minimal 2-zone, 1-week dense grid for characterization."""
        rng = np.random.RandomState(99)

        rows = []
        for zone_id in [1, 2]:
            for hour_offset in range(7 * 24):  # 1 week
                hour_ts = datetime(2023, 12, 25 + hour_offset // 24, hour_offset % 24)
                trip_count = zone_id * (hour_ts.hour + 1) + rng.randint(0, 5)
                rows.append(
                    {
                        "zone_id": zone_id,
                        "zone_name": f"Zone {zone_id}",
                        "hour_ts": hour_ts,
                        "trip_count": trip_count,
                        "avg_fare": 10.0 + rng.random() * 5,
                        "avg_distance": 1.0 + rng.random() * 3,
                    }
                )

        return pl.DataFrame(rows).cast(
            {
                "zone_id": pl.Int64,
                "trip_count": pl.Int64,
                "hour_ts": pl.Datetime("us"),
            }
        )

    def test_engineer_features_output_shape(self, small_dense_grid) -> None:
        result = engineer_features(small_dense_grid, lag_hours=[1, 24, 168], rolling_windows=[24])
        assert len(result) == len(small_dense_grid)
        assert "lag_1h" in result.columns
        assert "lag_24h" in result.columns
        assert "lag_168h" in result.columns
        assert "rolling_mean_24h" in result.columns
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "month" in result.columns

    def test_engineer_features_deterministic(self, small_dense_grid) -> None:
        """Two calls produce identical output."""
        r1 = engineer_features(small_dense_grid, lag_hours=[1, 24, 168], rolling_windows=[24])
        r2 = engineer_features(small_dense_grid, lag_hours=[1, 24, 168], rolling_windows=[24])
        assert r1.equals(r2)

    def test_engineer_features_lag_values(self, small_dense_grid) -> None:
        """Spot-check: lag_1h at row N equals trip_count at row N-1 within same zone."""
        result = engineer_features(small_dense_grid, lag_hours=[1], rolling_windows=[24])
        zone1 = result.filter(pl.col("zone_id") == 1).sort("hour_ts")
        assert zone1["lag_1h"][1] == float(zone1["trip_count"][0])
