"""Tests for data pipeline: schemas, grid completeness, splits, weekday convention,
schema enforcement with narrow dtypes, and atomic download safety."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from demandops.data.schemas import FeatureSchema, HourlyHistorySchema
from demandops.data.splits import temporal_split


class TestHourlyHistorySchema:
    def test_schema_validates(self, dense_history_df: pl.DataFrame) -> None:
        HourlyHistorySchema.validate(dense_history_df)

    def test_includes_december(self, dense_history_df: pl.DataFrame) -> None:
        dec_rows = dense_history_df.filter(pl.col("hour_ts") < datetime(2024, 1, 1))
        assert len(dec_rows) > 0

    def test_trip_count_non_negative(self, dense_history_df: pl.DataFrame) -> None:
        assert dense_history_df["trip_count"].min() >= 0

    def test_zone_name_populated(self, dense_history_df: pl.DataFrame) -> None:
        assert dense_history_df["zone_name"].null_count() == 0

    def test_zero_demand_hours_exist(self, dense_history_df: pl.DataFrame) -> None:
        zero_rows = dense_history_df.filter(pl.col("trip_count") == 0)
        assert len(zero_rows) > 0

    def test_validates_with_int32_zone_id(self, dense_history_df: pl.DataFrame) -> None:
        """Schema coercion handles DuckDB's Int32 output for zone_id."""
        df = dense_history_df.cast({"zone_id": pl.Int32, "trip_count": pl.Int32})
        HourlyHistorySchema.validate(df)


class TestGridCompleteness:
    def test_row_count(self, dense_history_df: pl.DataFrame, zone_universe: dict) -> None:
        n_zones = zone_universe["n_zones"]
        n_unique_hours = dense_history_df["hour_ts"].n_unique()
        expected = n_zones * n_unique_hours
        assert len(dense_history_df) == expected

    def test_every_zone_has_every_hour(self, dense_history_df: pl.DataFrame) -> None:
        counts = dense_history_df.group_by("zone_id").len()
        unique_counts = counts["len"].unique()
        assert len(unique_counts) == 1

    def test_no_duplicate_zone_hour_pairs(self, dense_history_df: pl.DataFrame) -> None:
        n_unique = dense_history_df.select(pl.struct("zone_id", "hour_ts").n_unique()).item()
        assert n_unique == len(dense_history_df)


class TestFeatureSchema:
    def test_schema_validates(self, features_df: pl.DataFrame) -> None:
        FeatureSchema.validate(features_df)

    def test_no_december_rows(self, features_df: pl.DataFrame) -> None:
        dec = features_df.filter(pl.col("hour_ts") < datetime(2024, 1, 1))
        assert len(dec) == 0

    def test_no_null_lags(self, features_df: pl.DataFrame) -> None:
        for col in ["lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h"]:
            assert features_df[col].null_count() == 0, f"{col} has nulls"

    def test_day_of_week_0_to_6(self, features_df: pl.DataFrame) -> None:
        """day_of_week must be 0–6 (0=Mon, 6=Sun), not 1–7."""
        assert features_df["day_of_week"].min() >= 0
        assert features_df["day_of_week"].max() <= 6

    def test_is_weekend_only_sat_sun(self, features_df: pl.DataFrame) -> None:
        """is_weekend=1 only when day_of_week >= 5 (Sat=5, Sun=6)."""
        weekend_rows = features_df.filter(pl.col("is_weekend") == 1)
        assert weekend_rows["day_of_week"].min() >= 5
        weekday_rows = features_df.filter(pl.col("is_weekend") == 0)
        assert weekday_rows["day_of_week"].max() <= 4

    def test_weekday_matches_python_convention(self, features_df: pl.DataFrame) -> None:
        """Spot-check: 2024-01-01 is a Monday → day_of_week should be 0."""
        monday_row = features_df.filter(
            (pl.col("hour_ts") == datetime(2024, 1, 1, 12, 0, 0)) & (pl.col("zone_id") == 1)
        )
        if len(monday_row) > 0:
            assert monday_row["day_of_week"].item() == 0  # Monday = 0

    def test_validates_with_narrow_dtypes(self, features_df: pl.DataFrame) -> None:
        """Schema coercion handles Polars' Int8 for hour_of_day/day_of_week/month."""
        df = features_df.cast(
            {
                "zone_id": pl.Int32,
                "trip_count": pl.Int32,
                "hour_of_day": pl.Int8,
                "day_of_week": pl.Int8,
                "is_weekend": pl.Int8,
                "month": pl.Int8,
            }
        )
        FeatureSchema.validate(df)


class TestTemporalSplit:
    def test_no_overlap_no_gap(self, features_df: pl.DataFrame, split_config: dict) -> None:
        train, val, test = temporal_split(features_df, **split_config["split"])
        # Exactly 1 hour gap between last train and first val timestamp
        gap = val["hour_ts"].min() - train["hour_ts"].max()
        assert gap == timedelta(hours=1)
        gap2 = test["hour_ts"].min() - val["hour_ts"].max()
        assert gap2 == timedelta(hours=1)

    def test_half_open_boundaries(self, features_df: pl.DataFrame, split_config: dict) -> None:
        train, val, test = temporal_split(features_df, **split_config["split"])
        assert train["hour_ts"].max() < datetime(2024, 2, 1)
        assert val["hour_ts"].min() >= datetime(2024, 2, 1)
        assert val["hour_ts"].max() < datetime(2024, 2, 15)
        assert test["hour_ts"].min() >= datetime(2024, 2, 15)
        assert test["hour_ts"].max() < datetime(2024, 3, 1)

    def test_chronological_order(self, features_df: pl.DataFrame, split_config: dict) -> None:
        train, val, test = temporal_split(features_df, **split_config["split"])
        assert train["hour_ts"].max() < val["hour_ts"].min()
        assert val["hour_ts"].max() < test["hour_ts"].min()

    def test_no_december_in_any_split(self, features_df: pl.DataFrame, split_config: dict) -> None:
        train, val, test = temporal_split(features_df, **split_config["split"])
        for name, split in [("train", train), ("val", val), ("test", test)]:
            dec = split.filter(pl.col("hour_ts") < datetime(2024, 1, 1))
            assert len(dec) == 0, f"{name} contains December rows"

    def test_test_includes_feb_29(self, features_df: pl.DataFrame, split_config: dict) -> None:
        _, _, test = temporal_split(features_df, **split_config["split"])
        feb29 = test.filter(
            (pl.col("hour_ts").dt.month() == 2) & (pl.col("hour_ts").dt.day() == 29)
        )
        assert len(feb29) > 0

    def test_splits_cover_all_data(self, features_df: pl.DataFrame, split_config: dict) -> None:
        train, val, test = temporal_split(features_df, **split_config["split"])
        total = len(train) + len(val) + len(test)
        assert total == len(features_df)


class TestAtomicDownload:
    def test_partial_download_leaves_no_file(self, tmp_path: Path) -> None:
        """Interrupted download must not leave a partial destination file."""
        from demandops.data.download import _atomic_download

        dest = tmp_path / "test.parquet"
        with patch(
            "demandops.data.download.urlretrieve",
            side_effect=ConnectionError("interrupted"),
        ):
            with pytest.raises(ConnectionError):
                _atomic_download("http://example.com/test.parquet", dest)

        assert not dest.exists(), "Partial destination file was left behind"
        tmp_file = dest.with_suffix(".parquet.tmp")
        assert not tmp_file.exists(), "Temp file was left behind"

    def test_successful_download_produces_dest(self, tmp_path: Path) -> None:
        """Successful download should produce the destination file atomically."""
        from demandops.data.download import _atomic_download

        dest = tmp_path / "test.csv"

        def fake_retrieve(url, path):
            Path(path).write_text("data")

        with patch("demandops.data.download.urlretrieve", side_effect=fake_retrieve):
            _atomic_download("http://example.com/test.csv", dest)

        assert dest.exists()
        assert dest.read_text() == "data"
        assert not dest.with_suffix(".csv.tmp").exists()
