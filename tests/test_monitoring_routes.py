"""Tests for monitoring API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demandops.db import get_db
from demandops.features import FEATURE_COLUMNS
from demandops.security.auth import RateLimiter, hash_key


@pytest.fixture
def monitoring_reference(tmp_path: Path) -> Path:
    """Minimal reference distributions for testing."""
    rng = np.random.RandomState(42)
    n = 5000
    ref = {"features": {}, "metadata": {"ks_subsample_size": n, "n_bins": 10}}

    for feature_name in FEATURE_COLUMNS:
        if feature_name == "zone_id":
            values = rng.choice([1, 2, 3], size=n).astype(float)
        elif feature_name == "hour_of_day":
            values = rng.choice(range(24), size=n).astype(float)
        elif feature_name == "day_of_week":
            values = rng.choice(range(7), size=n).astype(float)
        elif feature_name == "is_weekend":
            values = rng.choice([0, 1], size=n).astype(float)
        elif feature_name == "month":
            values = rng.choice(range(1, 13), size=n).astype(float)
        else:
            values = rng.exponential(5, size=n)

        quantiles = np.linspace(0, 100, 11)
        boundaries = np.percentile(values, quantiles).tolist()
        bin_counts = np.histogram(values, bins=boundaries)[0].tolist()
        ref["features"][feature_name] = {
            "decile_boundaries": boundaries,
            "bin_counts": bin_counts,
            "ks_subsample": values.tolist(),
        }

    cont_features = [c for c in FEATURE_COLUMNS if c != "zone_id"]
    cont_indices = [FEATURE_COLUMNS.index(c) for c in cont_features]
    full_matrix = np.column_stack(
        [np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS]
    )
    ref["correlation_matrix"] = np.corrcoef(full_matrix[:, cont_indices], rowvar=False).tolist()
    ref["correlation_features"] = cont_features

    path = tmp_path / "reference_distributions.json"
    path.write_text(json.dumps(ref))
    return path


@pytest.fixture
def monitoring_app(tmp_path: Path, monitoring_reference) -> FastAPI:
    from demandops.monitoring.drift_detector import DriftDetector
    from demandops.monitoring.quality_tracker import QualityTracker
    from demandops.serving.monitoring_routes import monitoring_router

    db = get_db(str(tmp_path / "monitoring_test.db"))
    detector = DriftDetector(monitoring_reference, min_samples=10)
    tracker = QualityTracker(db)

    app = FastAPI()
    app.state.db = db
    app.state.rate_limiter = RateLimiter()
    app.state.drift_detector = detector
    app.state.quality_tracker = tracker
    app.include_router(monitoring_router)

    # Create an API key for actuals endpoint
    raw_key = "monitoring-test-key"
    db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, is_active) "
        "VALUES (?, ?, ?, ?, ?)",
        (hash_key(raw_key), "test", "2024-01-01", 1000, True),
    )
    db.commit()
    app.state._test_api_key = raw_key

    return app


@pytest.fixture
def monitoring_client(monitoring_app) -> TestClient:
    return TestClient(monitoring_app)


class TestDriftEndpoint:
    def test_insufficient_samples(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_samples"
        assert data["collected"] == 0

    def test_returns_drift_after_samples(self, monitoring_app, monitoring_client) -> None:
        detector = monitoring_app.state.drift_detector
        rng = np.random.RandomState(42)
        ref_data = detector._reference
        for _ in range(50):
            vector = []
            for f in FEATURE_COLUMNS:
                vector.append(rng.choice(ref_data["features"][f]["ks_subsample"]))
            detector.accumulator.add(vector)

        resp = monitoring_client.get("/monitoring/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "warning")  # Small samples may show noise
        assert "features" in data
        assert "correlation_shift" in data

    def test_no_auth_required(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/drift")
        assert resp.status_code == 200  # No auth header, still works


class TestQualityEndpoint:
    def test_insufficient_pairs(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/quality")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_matched_pairs"

    def test_with_matched_data(self, monitoring_app, monitoring_client) -> None:
        tracker = monitoring_app.state.quality_tracker
        for i in range(10):
            pid = tracker.log_prediction(1, "2024-02-01T12:00:00", float(i * 10))
            tracker.submit_actuals([{"prediction_id": pid, "actual_value": float(i * 10 + 2)}])

        resp = monitoring_client.get("/monitoring/quality?window=7d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "mae" in data
        assert "rmse" in data
        assert "smape" in data

    def test_no_auth_required(self, monitoring_client) -> None:
        resp = monitoring_client.get("/monitoring/quality")
        assert resp.status_code == 200


class TestActualsEndpoint:
    def test_requires_auth(self, monitoring_client) -> None:
        resp = monitoring_client.post(
            "/monitoring/actuals",
            json={"actuals": [{"prediction_id": "abc", "actual_value": 10.0}]},
        )
        assert resp.status_code == 401

    def test_submit_actuals_with_auth(self, monitoring_app, monitoring_client) -> None:
        tracker = monitoring_app.state.quality_tracker
        pid = tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        resp = monitoring_client.post(
            "/monitoring/actuals",
            json={"actuals": [{"prediction_id": pid, "actual_value": 40.0}]},
            headers={"Authorization": f"Bearer {monitoring_app.state._test_api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched_count"] == 1
