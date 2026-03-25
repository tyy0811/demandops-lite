"""Tests for the serving API endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from prometheus_client import REGISTRY


@pytest.fixture(autouse=True)
def _reset_prometheus_metrics():
    """Reset Prometheus collector sample values between tests (fix #16).

    We can't unregister module-level collectors, but we can reset their
    internal state so counters don't leak across tests.
    """
    yield
    # Reset all sample values after each test
    for collector in list(REGISTRY._names_to_collectors.values()):
        if hasattr(collector, "_metrics"):
            collector._metrics.clear()


class TestPredictEndpoint:
    def test_valid_request(self, test_client) -> None:
        resp = test_client.post(
            "/predict",
            json={
                "zone_id": 1,
                "hour_ts": "2024-02-01T12:00:00",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["zone_id"] == 1
        assert data["predicted_count"] == 42.5
        assert data["model_name"] == "lightgbm"

    def test_unsupported_zone(self, test_client) -> None:
        resp = test_client.post(
            "/predict",
            json={
                "zone_id": 100,
                "hour_ts": "2024-02-01T12:00:00",
            },
        )
        assert resp.status_code == 422

    def test_december_timestamp(self, test_client) -> None:
        resp = test_client.post(
            "/predict",
            json={
                "zone_id": 1,
                "hour_ts": "2023-12-15T12:00:00",
            },
        )
        assert resp.status_code == 422

    def test_response_has_metadata(self, test_client) -> None:
        resp = test_client.post(
            "/predict",
            json={
                "zone_id": 1,
                "hour_ts": "2024-02-01T12:00:00",
            },
        )
        data = resp.json()
        assert "metadata" in data
        assert "latency_ms" in data["metadata"]
        assert "request_id" in data["metadata"]
        assert "features_used" in data["metadata"]


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client) -> None:
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_includes_n_supported_zones(self, test_client) -> None:
        data = test_client.get("/health").json()
        assert "n_supported_zones" in data
        assert data["n_supported_zones"] == 3

    def test_health_includes_supported_range(self, test_client) -> None:
        data = test_client.get("/health").json()
        assert "supported_start" in data
        assert "supported_end" in data


class TestMetricsEndpoint:
    def test_metrics_returns_text_plain(self, test_client) -> None:
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_prometheus_metrics(self, test_client) -> None:
        test_client.post(
            "/predict",
            json={
                "zone_id": 1,
                "hour_ts": "2024-02-01T12:00:00",
            },
        )
        body = test_client.get("/metrics").text
        assert "demandops_requests_total" in body
        assert "demandops_request_latency_seconds" in body


class TestDegradedHealth:
    @pytest.fixture
    def degraded_client_no_model(self, mock_feature_service):
        """App with feature service but no model artifact loaded."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from demandops.serving.routes import configure, router

        app = FastAPI()
        app.include_router(router)
        configure(
            app,
            mock_feature_service,
            None,
            "lightgbm",
            time.time(),
            model_artifact_loaded=False,
        )
        return TestClient(app)

    @pytest.fixture
    def degraded_client_no_services(self):
        """App with neither feature service nor model."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from demandops.serving.routes import configure, router

        app = FastAPI()
        app.include_router(router)
        configure(
            app,
            None,
            None,
            "lightgbm",
            time.time(),
            model_artifact_loaded=False,
        )
        return TestClient(app)

    def test_missing_model_reports_degraded(self, degraded_client_no_model) -> None:
        resp = degraded_client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["history_loaded"] is True

    def test_missing_everything_reports_degraded(self, degraded_client_no_services) -> None:
        resp = degraded_client_no_services.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["history_loaded"] is False
        assert data["supported_start"] is None
        assert data["supported_end"] is None
        assert data["n_supported_zones"] == 0

    def test_healthy_requires_both(self, test_client) -> None:
        """Healthy state requires both model artifact and feature service."""
        resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["history_loaded"] is True

    def test_create_app_missing_artifacts_enters_degraded(self, tmp_path) -> None:
        """Real create_app() with missing serving artifacts starts degraded."""
        import yaml
        from fastapi.testclient import TestClient
        from demandops.serving.app import create_app

        # Write a valid config pointing at nonexistent artifacts
        config = yaml.safe_load(Path("configs/default.yaml").read_text())
        config["serving"]["history_path"] = str(tmp_path / "missing.parquet")
        config["serving"]["feature_schema_path"] = str(tmp_path / "missing.json")
        config["serving"]["zone_universe_path"] = str(tmp_path / "missing.json")
        config["artifacts"]["models_dir"] = str(tmp_path / "no_models")

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(config))

        app = create_app(config_path=str(config_path))
        # Use context manager to ensure startup/shutdown lifecycle runs
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False
            assert data["history_loaded"] is False


class TestBatchPredictEndpoint:
    def test_batch_prediction(self, test_client) -> None:
        """Batch of 3 valid requests returns 3 predictions."""
        resp = test_client.post(
            "/predict/batch",
            json={
                "requests": [
                    {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                    {"zone_id": 2, "hour_ts": "2024-02-01T13:00:00"},
                    {"zone_id": 3, "hour_ts": "2024-02-01T14:00:00"},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction_count"] == 3
        assert len(data["predictions"]) == 3
        assert data["latency_ms"] > 0
        for pred in data["predictions"]:
            assert pred["predicted_count"] == 42.5
            assert pred["model_name"] == "lightgbm"

    def test_single_item_batch(self, test_client) -> None:
        """Batch of 1 item works (min_length=1)."""
        resp = test_client.post(
            "/predict/batch",
            json={
                "requests": [
                    {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["prediction_count"] == 1

    def test_empty_batch_rejected(self, test_client) -> None:
        """Empty batch returns 422 (min_length=1)."""
        resp = test_client.post(
            "/predict/batch",
            json={"requests": []},
        )
        assert resp.status_code == 422

    def test_batch_unsupported_zone_fails_all(self, test_client) -> None:
        """One bad zone_id in batch fails the entire request."""
        resp = test_client.post(
            "/predict/batch",
            json={
                "requests": [
                    {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                    {"zone_id": 999, "hour_ts": "2024-02-01T12:00:00"},
                ]
            },
        )
        assert resp.status_code == 422

    def test_batch_unsupported_timestamp_fails_all(self, test_client) -> None:
        """One bad timestamp in batch fails the entire request."""
        resp = test_client.post(
            "/predict/batch",
            json={
                "requests": [
                    {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                    {"zone_id": 1, "hour_ts": "2023-06-01T12:00:00"},
                ]
            },
        )
        assert resp.status_code == 422

    def test_batch_latency_covers_full_request(self, test_client) -> None:
        """latency_ms is present and positive."""
        resp = test_client.post(
            "/predict/batch",
            json={
                "requests": [
                    {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00"},
                ]
            },
        )
        assert resp.json()["latency_ms"] > 0
