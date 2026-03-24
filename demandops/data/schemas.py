"""Pandera data contracts for pipeline validation.

Weekday convention: 0=Mon, 6=Sun (Python datetime.weekday() convention).
Polars dt.weekday() returns 1–7; we subtract 1 before validation.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class HourlyHistorySchema(pa.DataFrameModel):
    """Schema for hourly_history.parquet — dense zone×hour grid.

    Covers Dec 2023 – Feb 2024. Every (zone_id, hour_ts) pair in the
    zone universe has exactly one row. trip_count is 0 for no-demand hours.
    avg_fare and avg_distance are nullable (null for zero-demand hours).
    """

    zone_id: int = pa.Field(ge=1, le=263)
    zone_name: str = pa.Field(nullable=False)
    hour_ts: pl.Datetime = pa.Field(nullable=False)
    trip_count: int = pa.Field(ge=0)
    avg_fare: float = pa.Field(nullable=True)
    avg_distance: float = pa.Field(nullable=True)


class FeatureSchema(pa.DataFrameModel):
    """Schema for features.parquet — model-ready features, Jan–Feb only.

    No December rows. All lags populated (dense grid guarantees this).
    No nulls in any column. day_of_week uses 0=Mon, 6=Sun convention.
    """

    zone_id: int = pa.Field(ge=1, le=263)
    hour_ts: pl.Datetime = pa.Field(nullable=False)
    trip_count: int = pa.Field(ge=0)
    hour_of_day: int = pa.Field(ge=0, le=23)
    day_of_week: int = pa.Field(ge=0, le=6)
    is_weekend: int = pa.Field(ge=0, le=1)
    month: int = pa.Field(ge=1, le=12)
    lag_1h: float = pa.Field(nullable=False)
    lag_24h: float = pa.Field(nullable=False)
    lag_168h: float = pa.Field(nullable=False)
    rolling_mean_24h: float = pa.Field(nullable=False)


class PredictionOutputSchema(pa.DataFrameModel):
    """Schema for prediction output validation."""

    zone_id: int = pa.Field(ge=1, le=263)
    hour_ts: pl.Datetime = pa.Field(nullable=False)
    predicted_count: float = pa.Field(ge=0.0)
