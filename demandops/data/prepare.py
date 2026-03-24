"""Data preparation pipeline: raw parquet → dense grid → features.

Pipeline steps:
1. Load 3 months of raw parquet via DuckDB
2. Filter nulls, negative fares, out-of-range dates
3. Determine zone universe, save as JSON
4. Aggregate to hourly trip counts per zone (DuckDB)
5. Densify grid: cartesian product of zones × hours (DuckDB)
6. Join zone lookup for zone names
7. Save hourly_history.parquet (Dec–Feb, dense)
8. Feature engineering with Polars (lags, temporal, rolling)
9. Drop December rows
10. Save features.parquet (Jan–Feb, model-ready)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import structlog

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
            pl.col("trip_count")
            .shift(lag)
            .over("zone_id")
            .cast(pl.Float64)
            .alias(f"lag_{lag}h")
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
                .rolling_mean(window_size=w, min_periods=w)
                .alias(f"rolling_mean_{w}h")
            )

        df = df.group_by("zone_id", maintain_order=True).map_groups(_add_rolling)

    return df


def prepare(
    raw_dir: Path,
    processed_dir: Path,
    zones_path: Path,
    zone_universe_path: Path,
    months: list[str],
    lag_hours: list[int],
    rolling_windows: list[int],
) -> dict[str, Path]:
    """Run full preparation pipeline.

    Returns:
        Dict with output paths: history_path, features_path, zone_universe_path
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    zone_universe_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Step 1: Load raw parquet files
    parquet_files = [str(raw_dir / f"yellow_tripdata_{m}.parquet") for m in months]
    logger.info("loading_raw_data", files=parquet_files)

    con.execute("""
        CREATE OR REPLACE TABLE raw_trips AS
        SELECT * FROM read_parquet(?)
    """, [parquet_files])

    raw_count = con.execute("SELECT COUNT(*) FROM raw_trips").fetchone()[0]
    logger.info("raw_rows_loaded", count=raw_count)

    # Step 2: Filter
    con.execute("""
        CREATE OR REPLACE TABLE filtered_trips AS
        SELECT *
        FROM raw_trips
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
          AND PULocationID IS NOT NULL
          AND PULocationID BETWEEN 1 AND 263
          AND total_amount IS NOT NULL
          AND total_amount >= 0
          AND trip_distance IS NOT NULL
          AND trip_distance >= 0
          AND tpep_pickup_datetime >= TIMESTAMP '2023-12-01 00:00:00'
          AND tpep_pickup_datetime < TIMESTAMP '2024-03-01 00:00:00'
    """)

    filtered_count = con.execute("SELECT COUNT(*) FROM filtered_trips").fetchone()[0]
    logger.info("filtered_rows", count=filtered_count, dropped=raw_count - filtered_count)

    # Step 3: Zone universe
    zone_ids_result = con.execute("""
        SELECT DISTINCT PULocationID AS zone_id
        FROM filtered_trips
        ORDER BY zone_id
    """).fetchall()
    zone_ids = [row[0] for row in zone_ids_result]

    zone_universe = {
        "zone_ids": zone_ids,
        "n_zones": len(zone_ids),
        "source": "raw_trips distinct PULocationIDs",
    }
    zone_universe_path.write_text(json.dumps(zone_universe, indent=2))
    logger.info("zone_universe_saved", n_zones=len(zone_ids), path=str(zone_universe_path))

    # Step 4: Aggregate to hourly
    con.execute("""
        CREATE OR REPLACE TABLE hourly_agg AS
        SELECT
            PULocationID AS zone_id,
            DATE_TRUNC('hour', tpep_pickup_datetime) AS hour_ts,
            COUNT(*)::INTEGER AS trip_count,
            AVG(total_amount) AS avg_fare,
            AVG(trip_distance) AS avg_distance
        FROM filtered_trips
        GROUP BY PULocationID, DATE_TRUNC('hour', tpep_pickup_datetime)
    """)

    # Step 5: Densify grid
    # Register zone_ids as a DuckDB value list
    zone_ids_sql = ", ".join(str(z) for z in zone_ids)
    con.execute(f"""
        CREATE OR REPLACE TABLE dense_grid AS
        WITH zones AS (
            SELECT unnest([{zone_ids_sql}]) AS zone_id
        ),
        hours AS (
            SELECT unnest(
                generate_series(
                    TIMESTAMP '2023-12-01 00:00:00',
                    TIMESTAMP '2024-02-29 23:00:00',
                    INTERVAL '1 hour'
                )
            ) AS hour_ts
        ),
        grid AS (
            SELECT z.zone_id, h.hour_ts
            FROM zones z
            CROSS JOIN hours h
        )
        SELECT
            g.zone_id,
            g.hour_ts,
            COALESCE(a.trip_count, 0)::INTEGER AS trip_count,
            a.avg_fare,
            a.avg_distance
        FROM grid g
        LEFT JOIN hourly_agg a
            ON g.zone_id = a.zone_id AND g.hour_ts = a.hour_ts
    """)

    grid_count = con.execute("SELECT COUNT(*) FROM dense_grid").fetchone()[0]
    logger.info("dense_grid_rows", count=grid_count)

    # Step 6: Join zone names
    con.execute(f"""
        CREATE OR REPLACE TABLE zones_lookup AS
        SELECT "LocationID" AS zone_id, "Zone" AS zone_name
        FROM read_csv('{zones_path}', auto_detect=true)
    """)

    con.execute("""
        CREATE OR REPLACE TABLE hourly_history AS
        SELECT
            g.zone_id,
            COALESCE(z.zone_name, 'Unknown') AS zone_name,
            g.hour_ts,
            g.trip_count,
            g.avg_fare,
            g.avg_distance
        FROM dense_grid g
        LEFT JOIN zones_lookup z ON g.zone_id = z.zone_id
        ORDER BY g.zone_id, g.hour_ts
    """)

    # Step 7: Validate and save hourly_history.parquet
    history_path = processed_dir / "hourly_history.parquet"
    history_df = con.execute("SELECT * FROM hourly_history").pl()
    history_df = HourlyHistorySchema.validate(history_df)
    logger.info("hourly_history_validated")
    history_df.write_parquet(history_path)
    logger.info("hourly_history_saved", path=str(history_path), rows=len(history_df))

    con.close()

    # Step 8: Feature engineering (Polars)
    features_df = engineer_features(history_df, lag_hours, rolling_windows)

    # Step 9: Drop December rows
    features_df = features_df.filter(pl.col("hour_ts") >= datetime(2024, 1, 1))

    # Drop rows with null lags or rolling means (shouldn't happen for Jan+
    # since Dec provides warm-up, but be explicit)
    lag_cols = [f"lag_{h}h" for h in lag_hours]
    rolling_cols = [f"rolling_mean_{w}h" for w in rolling_windows]
    features_df = features_df.drop_nulls(subset=lag_cols + rolling_cols)

    logger.info("features_after_cleanup", rows=len(features_df))

    # Step 10: Validate and save features.parquet
    features_path = processed_dir / "features.parquet"
    features_df = FeatureSchema.validate(features_df)
    logger.info("features_validated")
    features_df.write_parquet(features_path)
    logger.info("features_saved", path=str(features_path), rows=len(features_df))

    return {
        "history_path": history_path,
        "features_path": features_path,
        "zone_universe_path": zone_universe_path,
    }
