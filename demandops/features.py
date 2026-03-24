"""Shared feature definitions. Single source of truth for column order.

Every module that needs feature column names or indices imports from here.
If this list changes, all consumers update automatically.
"""

from __future__ import annotations

# Canonical feature column order. Used by:
# - training/train.py (extract from DataFrame)
# - training/evaluate.py (extract from DataFrame)
# - serving/feature_service.py (build feature dict)
# - serving/routes.py (build numpy array)
# - models/baselines.py (column index constants)
# - feature_schema.json (persisted artifact)
FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "zone_id",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
]

TARGET_COLUMN = "trip_count"

# Column indices for models that need positional access (baselines)
IDX_HOUR_OF_DAY = FEATURE_COLUMNS.index("hour_of_day")
IDX_DAY_OF_WEEK = FEATURE_COLUMNS.index("day_of_week")
IDX_IS_WEEKEND = FEATURE_COLUMNS.index("is_weekend")
IDX_MONTH = FEATURE_COLUMNS.index("month")
IDX_ZONE_ID = FEATURE_COLUMNS.index("zone_id")
IDX_LAG_1H = FEATURE_COLUMNS.index("lag_1h")
IDX_LAG_24H = FEATURE_COLUMNS.index("lag_24h")
IDX_LAG_168H = FEATURE_COLUMNS.index("lag_168h")
IDX_ROLLING_MEAN_24H = FEATURE_COLUMNS.index("rolling_mean_24h")
