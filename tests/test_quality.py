"""Tests for prediction quality monitoring."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from demandops.db import get_db


@pytest.fixture
def quality_db(tmp_path: Path):
    conn = get_db(str(tmp_path / "quality_test.db"))
    yield conn
    conn.close()


class TestPredictionLogging:
    def test_log_returns_prediction_id(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid = tracker.log_prediction(zone_id=1, hour_ts="2024-02-01T12:00:00", predicted_value=42.5)
        assert isinstance(pid, str)
        assert len(pid) == 36  # UUID format

    def test_logged_prediction_in_db(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid = tracker.log_prediction(zone_id=1, hour_ts="2024-02-01T12:00:00", predicted_value=42.5)
        row = quality_db.execute(
            "SELECT zone_id, predicted_value, actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid,),
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 42.5
        assert row[2] is None  # No actual yet

    def test_concurrent_logging_no_errors(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        errors = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    tracker.log_prediction(
                        zone_id=thread_id,
                        hour_ts=f"2024-02-01T{i:02d}:00:00",
                        predicted_value=float(i),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes failed: {errors}"
        count = quality_db.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
        assert count == 200


class TestActualsSubmission:
    def test_match_by_prediction_id(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid = tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        result = tracker.submit_actuals(
            [
                {"prediction_id": pid, "actual_value": 40.0},
            ]
        )
        assert result["matched_count"] == 1
        assert result["unmatched_count"] == 0

        row = quality_db.execute(
            "SELECT actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid,),
        ).fetchone()
        assert row[0] == 40.0

    def test_match_by_zone_ts(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        result = tracker.submit_actuals(
            [
                {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00", "actual_value": 40.0},
            ]
        )
        assert result["matched_count"] == 1

    def test_ambiguous_match_uses_most_recent(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pid1 = tracker.log_prediction(1, "2024-02-01T12:00:00", 30.0)
        pid2 = tracker.log_prediction(1, "2024-02-01T12:00:00", 42.5)

        tracker.submit_actuals(
            [
                {"zone_id": 1, "hour_ts": "2024-02-01T12:00:00", "actual_value": 40.0},
            ]
        )

        # Most recent (pid2) should be matched
        row1 = quality_db.execute(
            "SELECT actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid1,),
        ).fetchone()
        row2 = quality_db.execute(
            "SELECT actual_value FROM prediction_log WHERE prediction_id = ?",
            (pid2,),
        ).fetchone()
        assert row1[0] is None
        assert row2[0] == 40.0

    def test_unmatched_returns_warning(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        result = tracker.submit_actuals(
            [
                {"prediction_id": "nonexistent-id", "actual_value": 40.0},
            ]
        )
        assert result["unmatched_count"] == 1
        assert len(result["warnings"]) == 1


class TestQualityComputation:
    def _seed_matched_pairs(self, tracker, pairs: list[tuple[float, float]]) -> None:
        """Helper: log predictions and submit actuals for known pairs."""
        pids = []
        for pred, _ in pairs:
            pid = tracker.log_prediction(1, "2024-02-01T12:00:00", pred)
            pids.append(pid)
        actuals = [
            {"prediction_id": pid, "actual_value": actual} for pid, (_, actual) in zip(pids, pairs)
        ]
        tracker.submit_actuals(actuals)

    def test_mae_computation(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pairs = [
            (10, 12),
            (20, 18),
            (30, 33),
            (40, 38),
            (50, 52),
            (60, 58),
            (70, 73),
            (80, 78),
            (90, 92),
            (100, 98),
        ]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        assert result["status"] == "ok"

        expected_mae = np.mean([abs(p - a) for p, a in pairs])
        assert abs(result["mae"] - expected_mae) < 0.001

    def test_rmse_computation(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pairs = [
            (10, 12),
            (20, 18),
            (30, 33),
            (40, 38),
            (50, 52),
            (60, 58),
            (70, 73),
            (80, 78),
            (90, 92),
            (100, 98),
        ]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        expected_rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in pairs]))
        assert abs(result["rmse"] - expected_rmse) < 0.001

    def test_smape_bounded_with_zeros(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        # Include zero actuals that would make MAPE infinite
        pairs = [
            (10, 12),
            (5, 0),
            (0, 0),
            (20, 18),
            (30, 33),
            (40, 38),
            (50, 52),
            (60, 58),
            (70, 73),
            (80, 78),
        ]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        assert result["status"] == "ok"
        assert result["smape"] < 200  # sMAPE is bounded [0, 200]
        assert result["zero_denominator_pairs_excluded"] >= 1  # (0, 0) pair

    def test_insufficient_pairs_gate(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)
        pairs = [(10, 12), (20, 18)]
        self._seed_matched_pairs(tracker, pairs)

        result = tracker.compute_quality(window="7d")
        assert result["status"] == "insufficient_matched_pairs"
        assert result["matched"] == 2
        assert result["required"] == 10

    def test_window_filtering(self, quality_db) -> None:
        from demandops.monitoring.quality_tracker import QualityTracker

        tracker = QualityTracker(quality_db)

        # Log 10 old predictions (outside window) manually
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        for i in range(10):
            quality_db.execute(
                "INSERT INTO prediction_log (prediction_id, zone_id, hour_ts, predicted_value, actual_value, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"old-{i}", 1, "2024-01-01T12:00:00", 10.0, 12.0, old_time),
            )
        quality_db.commit()

        # Log 5 recent predictions (inside window)
        pairs = [(10, 12), (20, 18), (30, 33), (40, 38), (50, 52)]
        self._seed_matched_pairs(tracker, pairs)

        # window=1d should only see the 5 recent ones
        result = tracker.compute_quality(window="1d")
        assert result["status"] == "insufficient_matched_pairs"
        assert result["matched"] == 5
