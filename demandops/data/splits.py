"""Temporal split using half-open intervals.

Every boundary uses >= start and < end. No ambiguity.
train_end == val_start, val_end == test_start: no gaps, no overlaps.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl


def temporal_split(
    df: pl.DataFrame,
    train_start: str | datetime,
    train_end: str | datetime,
    val_end: str | datetime,
    test_end: str | datetime,
    ts_column: str = "hour_ts",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame into train/val/test using half-open intervals.

    Args:
        df: DataFrame with a timestamp column
        train_start: Inclusive start of training window
        train_end: Exclusive end of training / inclusive start of validation
        val_end: Exclusive end of validation / inclusive start of test
        test_end: Exclusive end of test window
        ts_column: Name of the timestamp column

    Returns:
        (train, val, test) DataFrames
    """
    if isinstance(train_start, str):
        train_start = datetime.fromisoformat(train_start)
    if isinstance(train_end, str):
        train_end = datetime.fromisoformat(train_end)
    if isinstance(val_end, str):
        val_end = datetime.fromisoformat(val_end)
    if isinstance(test_end, str):
        test_end = datetime.fromisoformat(test_end)

    ts = pl.col(ts_column)

    train = df.filter((ts >= train_start) & (ts < train_end))
    val = df.filter((ts >= train_end) & (ts < val_end))
    test = df.filter((ts >= val_end) & (ts < test_end))

    return train, val, test


def split_from_config(
    df: pl.DataFrame, config: dict, ts_column: str = "hour_ts"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split using config dict with split.train_start etc."""
    split_cfg = config["split"]
    return temporal_split(
        df,
        train_start=split_cfg["train_start"],
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
        test_end=split_cfg["test_end"],
        ts_column=ts_column,
    )
