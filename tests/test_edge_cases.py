"""Edge-case tests for the serving layer."""

from __future__ import annotations


class TestEdgeCases:

    def test_zone_id_zero(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 0,
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_zone_id_264(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 264,
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_missing_zone_id(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_missing_hour_ts(self, test_client) -> None:
        resp = test_client.post("/predict", json={"zone_id": 1})
        assert resp.status_code == 422

    def test_invalid_hour_ts(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "not-a-date",
        })
        assert resp.status_code == 422

    def test_feb_29_2024_valid(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-02-29T12:00:00",
        })
        assert resp.status_code == 200

    def test_boundary_supported_start(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-01-01T00:00:00",
        })
        assert resp.status_code == 200

    def test_boundary_supported_end_exclusive(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-03-01T00:00:00",
        })
        assert resp.status_code == 422
