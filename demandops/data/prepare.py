"""Data preparation pipeline: adapter -> dense grid -> features.

Pipeline steps:
1. Adapter.prepare_hourly_history() produces dense grid (dataset-specific)
2. Validate with HourlyHistorySchema
3. Save hourly_history.parquet
4. Feature engineering with Polars (lags, temporal, rolling) -- shared
5. Drop warm-up rows
6. Validate with FeatureSchema
7. Save features.parquet
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog

from demandops.data.adapters.base import DatasetAdapter
from demandops.data.schemas import FeatureSchema, HourlyHistorySchema

logger = structlog.get_logger()


def engineer_features(
    df: pl.DataFrame,
    lag_hours: list[int],
    rolling_windows: list[int],
) -> pl.DataFrame:
    """Add temporal features, lags, and rolling means.

    Operates on the dense grid so all lags are guaranteed non-null
    after the warm-up period.

    IMPORTANT: Polars dt.weekday() returns 1=Mon through 7=Sun.
    We subtract 1 to get 0=Mon through 6=Sun, matching Python's
    datetime.weekday() convention. This ensures train-serve parity
    with FeatureService which uses Python datetime.
    """
    # Must be sorted per zone for correct lag/rolling computation
    df = df.sort(["zone_id", "hour_ts"])

    # Temporal features (fix #1: subtract 1 from Polars weekday)
    polars_weekday = pl.col("hour_ts").dt.weekday() - 1  # 0=Mon, 6=Sun
    df = df.with_columns(
        pl.col("hour_ts").dt.hour().alias("hour_of_day"),
        polars_weekday.alias("day_of_week"),
        (polars_weekday >= 5).cast(pl.Int64).alias("is_weekend"),  # fix #3: Sat=5, Sun=6
        pl.col("hour_ts").dt.month().alias("month"),
    )

    # Lag features (per zone, using shift on sorted data)
    for lag in lag_hours:
        df = df.with_columns(
            pl.col("trip_count").shift(lag).over("zone_id").cast(pl.Float64).alias(f"lag_{lag}h")
        )

    # Rolling mean (fix #2: group_by.map_groups for unambiguous per-group rolling)
    # We need: for each row, the mean of trip_count at hours [t-24, t-1]
    # Using map_groups because rolling_mean().over() has version-dependent
    # behavior — rolling is a window function that operates on physical row
    # positions, and .over() partitioning doesn't guarantee correct boundaries.
    for window in rolling_windows:

        def _add_rolling(group_df: pl.DataFrame, w: int = window) -> pl.DataFrame:
            return group_df.with_columns(
                pl.col("trip_count")
                .shift(1)
                .rolling_mean(window_size=w, min_samples=w)
                .alias(f"rolling_mean_{w}h")
            )

        df = df.group_by("zone_id", maintain_order=True).map_groups(_add_rolling)

    return df


def prepare(
    adapter: DatasetAdapter,
    raw_dir: Path,
    processed_dir: Path,
    zone_universe_path: Path,
    config: dict,
) -> dict[str, Path]:
    """Run full preparation pipeline using the given adapter.

    Returns:
        Dict with output paths: history_path, features_path, zone_universe_path
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    zone_universe_path.parent.mkdir(parents=True, exist_ok=True)

    lag_hours = config["features"]["lag_hours"]
    rolling_windows = config["features"]["rolling_windows"]

    # Step 1: Adapter produces dense hourly history
    history_df, zone_ids = adapter.prepare_hourly_history(raw_dir, processed_dir, config)

    # Save zone universe
    zone_universe = {
        "zone_ids": zone_ids,
        "n_zones": len(zone_ids),
        "source": f"{adapter.name} adapter",
    }
    zone_universe_path.write_text(json.dumps(zone_universe, indent=2))
    logger.info("zone_universe_saved", n_zones=len(zone_ids), path=str(zone_universe_path))

    # Step 2: Validate hourly history
    history_df = HourlyHistorySchema.validate(history_df)
    logger.info("hourly_history_validated")

    # Step 3: Save hourly_history.parquet
    history_path = processed_dir / "hourly_history.parquet"
    history_df.write_parquet(history_path)
    logger.info("hourly_history_saved", path=str(history_path), rows=len(history_df))

    # Step 4: Feature engineering (shared)
    features_df = engineer_features(history_df, lag_hours, rolling_windows)

    # Step 5: Drop warm-up rows
    train_start = datetime.fromisoformat(config["split"]["train_start"])
    features_df = features_df.filter(pl.col("hour_ts") >= train_start)

    lag_cols = [f"lag_{h}h" for h in lag_hours]
    rolling_cols = [f"rolling_mean_{w}h" for w in rolling_windows]
    features_df = features_df.drop_nulls(subset=lag_cols + rolling_cols)
    logger.info("features_after_cleanup", rows=len(features_df))

    # Step 6: Validate features
    features_df = FeatureSchema.validate(features_df)
    logger.info("features_validated")

    # Step 7: Save features.parquet
    features_path = processed_dir / "features.parquet"
    features_df.write_parquet(features_path)
    logger.info("features_saved", path=str(features_path), rows=len(features_df))

    return {
        "history_path": history_path,
        "features_path": features_path,
        "zone_universe_path": zone_universe_path,
    }
