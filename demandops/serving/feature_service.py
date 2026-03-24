"""FeatureService: reconstruct lag features at request time from dense history.

Uses Python datetime.weekday() (0=Mon, 6=Sun) for consistency with
the training pipeline which normalizes Polars weekday to the same convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import structlog

from demandops.features import FEATURE_COLUMNS

logger = structlog.get_logger()


@dataclass
class FeatureResult:
    features: dict | None
    supported: bool
    warnings: list[str] = field(default_factory=list)


class FeatureService:
    """Serves features for prediction requests.

    Loads the dense hourly history grid and reconstructs lag features
    at request time, ensuring train-serve parity.
    """

    def __init__(
        self,
        history_path: Path,
        schema_path: Path,
        zone_universe_path: Path,
        config: dict,
    ) -> None:
        self.history = pl.read_parquet(history_path)
        self.schema = json.loads(Path(schema_path).read_text())
        zone_data = json.loads(Path(zone_universe_path).read_text())
        self.zone_universe: set[int] = set(zone_data["zone_ids"])

        # Use persisted feature schema to determine column order.
        # This is the artifact saved during training — if it diverges
        # from FEATURE_COLUMNS in code, we fail loudly at startup.
        self._feature_columns: list[str] = self.schema["columns"]
        if self._feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                f"Persisted feature schema {schema_path} column order "
                f"{self._feature_columns} does not match code constant "
                f"FEATURE_COLUMNS {FEATURE_COLUMNS}. Retrain or update code."
            )

        self._min_history_ts: datetime = self.history["hour_ts"].min()
        self._max_history_ts: datetime = self.history["hour_ts"].max()
        self._train_start = datetime.fromisoformat(config["split"]["train_start"])

        # Build lookup: (zone_id, hour_ts) → trip_count
        self._lookup: dict[tuple[int, datetime], int] = {}
        self._zone_names: dict[int, str] = {}
        for row in self.history.iter_rows(named=True):
            key = (row["zone_id"], row["hour_ts"])
            self._lookup[key] = row["trip_count"]
            if row["zone_id"] not in self._zone_names:
                self._zone_names[row["zone_id"]] = row["zone_name"]

        logger.info(
            "feature_service_loaded",
            history_rows=len(self.history),
            n_zones=len(self.zone_universe),
            supported_start=str(self.supported_start),
            supported_end=str(self.supported_end),
        )

    @property
    def supported_start(self) -> datetime:
        return self._train_start

    @property
    def supported_end(self) -> datetime:
        return self._max_history_ts + timedelta(hours=1)

    @property
    def n_supported_zones(self) -> int:
        return len(self.zone_universe)

    def get_zone_name(self, zone_id: int) -> str:
        return self._zone_names.get(zone_id, f"Unknown Zone {zone_id}")

    def get_features(self, zone_id: int, hour_ts: datetime) -> FeatureResult:
        # Normalize to naive UTC for consistent lookup (fix #15)
        # Pydantic may parse "2024-02-01T12:00:00+02:00" as timezone-aware;
        # we must convert to UTC first, then strip tzinfo for lookup.
        if hour_ts.tzinfo is not None:
            hour_ts = hour_ts.astimezone(timezone.utc).replace(tzinfo=None)

        warnings: list[str] = []

        if zone_id not in self.zone_universe:
            return FeatureResult(
                features=None,
                supported=False,
                warnings=[f"zone_id {zone_id} not in supported zone universe"],
            )

        if hour_ts < self.supported_start or hour_ts >= self.supported_end:
            return FeatureResult(
                features=None,
                supported=False,
                warnings=[
                    f"hour_ts {hour_ts.isoformat()} outside supported range "
                    f"[{self.supported_start.isoformat()}, "
                    f"{self.supported_end.isoformat()})"
                ],
            )

        # Temporal features (Python weekday: 0=Mon, 6=Sun)
        day_of_week = hour_ts.weekday()

        # Lag features from dense history
        lag_1h = self._get_trip_count(zone_id, hour_ts - timedelta(hours=1))
        lag_24h = self._get_trip_count(zone_id, hour_ts - timedelta(hours=24))
        lag_168h = self._get_trip_count(zone_id, hour_ts - timedelta(hours=168))

        # Rolling mean 24h: mean of trip_count at hours [t-24, t-1]
        rolling_vals = []
        for offset in range(1, 25):
            val = self._get_trip_count(zone_id, hour_ts - timedelta(hours=offset))
            if val is not None:
                rolling_vals.append(val)
        rolling_mean_24h = sum(rolling_vals) / len(rolling_vals) if rolling_vals else 0.0

        # Build features dict in FEATURE_COLUMNS order
        features = {
            "hour_of_day": hour_ts.hour,
            "day_of_week": day_of_week,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "month": hour_ts.month,
            "zone_id": zone_id,
            "lag_1h": float(lag_1h) if lag_1h is not None else 0.0,
            "lag_24h": float(lag_24h) if lag_24h is not None else 0.0,
            "lag_168h": float(lag_168h) if lag_168h is not None else 0.0,
            "rolling_mean_24h": rolling_mean_24h,
        }

        # Verify key order matches persisted feature schema
        assert list(features.keys()) == self._feature_columns

        return FeatureResult(features=features, supported=True, warnings=warnings)

    def _get_trip_count(self, zone_id: int, hour_ts: datetime) -> int | None:
        return self._lookup.get((zone_id, hour_ts))
