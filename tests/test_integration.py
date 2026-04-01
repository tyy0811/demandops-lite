"""Integration tests: auth + prediction + drift accumulation + quality logging."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from demandops.db import get_db
from demandops.features import FEATURE_COLUMNS
from demandops.security.auth import RateLimiter, hash_key


@pytest.fixture
def integration_app(
    mock_feature_service, mock_model, tmp_path: Path
) -> tuple[FastAPI, str]:
    """Full app with auth, drift detector, and quality tracker."""
    from demandops.monitoring.drift_detector import DriftDetector
    from demandops.monitoring.quality_tracker import QualityTracker
    from demandops.serving.monitoring_routes import monitoring_router
    from demandops.serving.routes import configure, router

    # Build reference distributions
    rng = np.random.RandomState(42)
    n = 5000
    ref = {"features": {}, "metadata": {"ks_subsample_size": n, "n_bins": 10}}
    for f in FEATURE_COLUMNS:
        if f in ("zone_id", "hour_of_day", "day_of_week", "is_weekend", "month"):
            values = rng.choice(range(24), size=n).astype(float)
        else:
            values = rng.exponential(5, size=n)
        quantiles = np.linspace(0, 100, 11)
        boundaries = np.percentile(values, quantiles).tolist()
        ref["features"][f] = {
            "decile_boundaries": boundaries,
            "bin_counts": np.histogram(values, bins=boundaries)[0].tolist(),
            "ks_subsample": values.tolist(),
        }
    cont = [c for c in FEATURE_COLUMNS if c != "zone_id"]
    full = np.column_stack([
        np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS
    ])
    ref["correlation_matrix"] = np.corrcoef(
        full[:, [FEATURE_COLUMNS.index(c) for c in cont]], rowvar=False
    ).tolist()
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref))

    db = get_db(str(tmp_path / "integration.db"))
    raw_key = "integration-test-key"
    db.execute(
        "INSERT INTO api_keys (key_hash, client_name, created_at, rate_limit, is_active) "
        "VALUES (?, ?, ?, ?, ?)",
        (hash_key(raw_key), "integration", "2024-01-01", 1000, True),
    )
    db.commit()

    app = FastAPI()
    app.state.db = db
    app.state.rate_limiter = RateLimiter()
    app.state.drift_detector = DriftDetector(ref_path, min_samples=5)
    app.state.quality_tracker = QualityTracker(db)
    app.include_router(router)
    app.include_router(monitoring_router)

    configure(
        app, mock_feature_service, mock_model, "lightgbm", time.time(),
        model_artifact_loaded=True, model_objective="regression",
        model_version="lightgbm-regression",
    )

    return app, raw_key


class TestPredictionToDriftPipeline:
    def test_predictions_accumulate_in_drift_detector(self, integration_app) -> None:
        app, key = integration_app
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {key}"}

        for _ in range(10):
            resp = client.post(
                "/predict",
                json={"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                headers=headers,
            )
            assert resp.status_code == 200

        drift_resp = client.get("/monitoring/drift")
        data = drift_resp.json()
        assert data["collected"] >= 5  # min_samples=5, should have 10

    def test_predictions_logged_to_quality_tracker(self, integration_app) -> None:
        app, key = integration_app
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {key}"}

        resp = client.post(
            "/predict",
            json={"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
            headers=headers,
        )
        prediction_id = resp.json()["prediction_id"]

        # Submit actual
        actuals_resp = client.post(
            "/monitoring/actuals",
            json={"actuals": [{"prediction_id": prediction_id, "actual_value": 40.0}]},
            headers=headers,
        )
        assert actuals_resp.json()["matched_count"] == 1

    def test_batch_predictions_logged(self, integration_app) -> None:
        app, key = integration_app
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {key}"}

        resp = client.post(
            "/predict/batch",
            json={"requests": [
                {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                {"zone_id": 2, "hour_ts": "2024-02-01T13:00:00"},
            ]},
            headers=headers,
        )
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) == 2

        # Both should have valid prediction_ids matchable via actuals
        for pred in preds:
            actuals_resp = client.post(
                "/monitoring/actuals",
                json={"actuals": [{"prediction_id": pred["prediction_id"], "actual_value": 40.0}]},
                headers=headers,
            )
            assert actuals_resp.json()["matched_count"] == 1
