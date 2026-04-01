"""Shared test fixtures. Deterministic, dense, synthetic data.

Uses engineer_features() from prepare.py — no duplicated logic.

Serving fixtures use lazy imports (fix #9) so that Day 1 tests
can run before the serving modules exist. The imports execute only
when a test actually requests the fixture.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from demandops.data.prepare import engineer_features

SEED = 42


# ──────────────────────────────────────────────
# Data pipeline fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def zone_universe() -> dict:
    """Small zone universe for testing: 3 zones."""
    return {"zone_ids": [1, 2, 3], "n_zones": 3, "source": "test fixture"}


@pytest.fixture
def zone_universe_path(tmp_path: Path, zone_universe: dict) -> Path:
    path = tmp_path / "zone_universe.json"
    path.write_text(json.dumps(zone_universe))
    return path


@pytest.fixture
def dense_history_df(zone_universe: dict) -> pl.DataFrame:
    """Dense zone×hour grid: 3 zones × Dec 2023 – Feb 2024.

    Deterministic trip_count: zone_id * hour_of_day + day_of_week.
    Zone 3 has zero demand late night (hours 0-4).
    """
    rng = np.random.RandomState(SEED)
    zone_ids = zone_universe["zone_ids"]

    start = datetime(2023, 12, 1, 0, 0, 0)
    end = datetime(2024, 2, 29, 23, 0, 0)  # 2024 is leap year

    hours = []
    ts = start
    while ts <= end:
        hours.append(ts)
        ts += timedelta(hours=1)

    rows = []
    for zone_id in zone_ids:
        for hour_ts in hours:
            # Deterministic demand pattern using Python weekday (0=Mon)
            base = zone_id * hour_ts.hour + hour_ts.weekday()
            if zone_id == 3 and hour_ts.hour < 5:
                trip_count = 0
                avg_fare = None
                avg_distance = None
            else:
                trip_count = max(0, base + rng.randint(-2, 5))
                avg_fare = float(round(10.0 + trip_count * 0.5 + rng.random() * 5, 2))
                avg_distance = float(round(1.0 + rng.random() * 3, 2))

            rows.append(
                {
                    "zone_id": zone_id,
                    "zone_name": f"Zone {zone_id}",
                    "hour_ts": hour_ts,
                    "trip_count": trip_count,
                    "avg_fare": avg_fare,
                    "avg_distance": avg_distance,
                }
            )

    return pl.DataFrame(rows).cast(
        {
            "zone_id": pl.Int64,
            "trip_count": pl.Int64,
            "hour_ts": pl.Datetime("us"),
        }
    )


@pytest.fixture
def features_df(dense_history_df: pl.DataFrame) -> pl.DataFrame:
    """Features DataFrame: Jan–Feb only, with lags and temporal features.

    Uses engineer_features() from prepare.py — no duplicated logic (fix #5).
    """
    df = engineer_features(dense_history_df, lag_hours=[1, 24, 168], rolling_windows=[24])

    # Drop December rows
    df = df.filter(pl.col("hour_ts") >= datetime(2024, 1, 1))

    # Drop rows with null lags (warm-up period)
    df = df.drop_nulls(subset=["lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h"])

    return df


@pytest.fixture
def split_config() -> dict:
    """Split configuration matching default.yaml."""
    return {
        "split": {
            "train_start": "2024-01-01T00:00:00",
            "train_end": "2024-02-01T00:00:00",
            "val_end": "2024-02-15T00:00:00",
            "test_end": "2024-03-01T00:00:00",
        }
    }


@pytest.fixture
def history_parquet_path(tmp_path: Path, dense_history_df: pl.DataFrame) -> Path:
    path = tmp_path / "hourly_history.parquet"
    dense_history_df.write_parquet(path)
    return path


# ──────────────────────────────────────────────
# Serving fixtures (fix #9: lazy imports — no top-level serving deps)
# All serving module imports are inside fixture bodies so that
# Day 1 pytest collection succeeds before serving code exists.
# ──────────────────────────────────────────────


@pytest.fixture
def feature_schema_path(tmp_path: Path) -> Path:
    from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN

    schema = {
        "columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "categorical": ["zone_id"],
        "numerical": [c for c in FEATURE_COLUMNS if c != "zone_id"],
    }
    path = tmp_path / "feature_schema.json"
    path.write_text(json.dumps(schema))
    return path


@pytest.fixture
def feature_service(
    history_parquet_path: Path,
    feature_schema_path: Path,
    zone_universe_path: Path,
    split_config: dict,
):
    from demandops.serving.feature_service import FeatureService

    return FeatureService(
        history_path=history_parquet_path,
        schema_path=feature_schema_path,
        zone_universe_path=zone_universe_path,
        config=split_config,
    )


@pytest.fixture
def mock_feature_service(zone_universe: dict) -> MagicMock:
    """Mock FeatureService with predictable responses.

    Uses a local FeatureResult dataclass to avoid importing from
    serving module at fixture definition time.
    """
    from demandops.serving.feature_service import FeatureResult

    svc = MagicMock()
    svc.supported_start = datetime(2024, 1, 1)
    svc.supported_end = datetime(2024, 3, 1)
    svc.n_supported_zones = zone_universe["n_zones"]
    svc.zone_universe = set(zone_universe["zone_ids"])
    svc.history = MagicMock()
    svc.history.__len__ = MagicMock(return_value=6624)
    svc.get_zone_name.return_value = "Test Zone"

    def get_features(zone_id, hour_ts):
        if zone_id not in svc.zone_universe:
            return FeatureResult(
                features=None,
                supported=False,
                warnings=[f"zone_id {zone_id} not in supported zone universe"],
            )
        if hour_ts < svc.supported_start or hour_ts >= svc.supported_end:
            return FeatureResult(
                features=None,
                supported=False,
                warnings=["hour_ts outside supported range"],
            )
        return FeatureResult(
            features={
                "hour_of_day": hour_ts.hour,
                "day_of_week": hour_ts.weekday(),  # Python: 0=Mon
                "is_weekend": 1 if hour_ts.weekday() >= 5 else 0,
                "month": hour_ts.month,
                "zone_id": zone_id,
                "lag_1h": 5.0,
                "lag_24h": 4.0,
                "lag_168h": 6.0,
                "rolling_mean_24h": 5.0,
            },
            supported=True,
        )

    svc.get_features.side_effect = get_features
    return svc


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.predict.side_effect = lambda X: np.full(len(X), 42.5)
    return model


@pytest.fixture
def test_db(tmp_path: Path):
    from demandops.db import get_db

    conn = get_db(str(tmp_path / "test.db"))
    yield conn
    conn.close()


@pytest.fixture
def api_key(test_db) -> str:
    """Create a test API key, return the raw key."""
    from demandops.security.auth import hash_key

    raw_key = "test-api-key-for-unit-tests-1234567890"
    key_hash = hash_key(raw_key)
    test_db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, max_batch_size, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (key_hash, "test_client", "2024-01-01T00:00:00", 1000, 10000, True),
    )
    test_db.commit()
    return raw_key


@pytest.fixture
def test_app(mock_feature_service, mock_model, test_db, api_key):
    from fastapi import FastAPI
    from demandops.security.auth import RateLimiter
    from demandops.serving.routes import configure, router

    app = FastAPI()
    app.include_router(router)
    app.state.db = test_db
    app.state.rate_limiter = RateLimiter()
    configure(
        app,
        mock_feature_service,
        mock_model,
        "lightgbm",
        time.time(),
        model_artifact_loaded=True,
        model_objective="regression",
        model_version="lightgbm-regression",
    )
    return app


@pytest.fixture
def test_client(test_app, api_key):
    from fastapi.testclient import TestClient

    client = TestClient(test_app)
    client.headers["Authorization"] = f"Bearer {api_key}"
    return client
