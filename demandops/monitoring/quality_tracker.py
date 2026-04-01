"""Prediction quality monitoring: log predictions, match actuals, compute metrics.

Predictions are logged to SQLite with a UUID. When ground truth arrives
(with a lag), actuals are matched by prediction_id or (zone_id, hour_ts).
Quality metrics (MAE, RMSE, sMAPE) computed over a rolling window.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np


class QualityTracker:
    """Logs predictions and computes quality metrics against actuals."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        self._lock = threading.Lock()

    def log_prediction(self, zone_id: int, hour_ts: str, predicted_value: float) -> str:
        """Log a prediction. Returns the prediction_id (UUID)."""
        prediction_id = str(uuid.uuid4())
        with self._lock:
            self._db.execute(
                "INSERT INTO prediction_log "
                "(prediction_id, zone_id, hour_ts, predicted_value, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    prediction_id,
                    zone_id,
                    hour_ts,
                    predicted_value,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._db.commit()
        return prediction_id

    def submit_actuals(self, actuals: list[dict]) -> dict:
        """Match actuals to logged predictions. Returns match summary."""
        matched = 0
        unmatched = 0
        warnings: list[str] = []

        with self._lock:
            for actual in actuals:
                if "prediction_id" in actual and actual["prediction_id"] is not None:
                    row = self._db.execute(
                        "SELECT prediction_id FROM prediction_log WHERE prediction_id = ?",
                        (actual["prediction_id"],),
                    ).fetchone()
                    if row:
                        self._db.execute(
                            "UPDATE prediction_log SET actual_value = ? WHERE prediction_id = ?",
                            (actual["actual_value"], actual["prediction_id"]),
                        )
                        matched += 1
                    else:
                        unmatched += 1
                        warnings.append(f"prediction_id {actual['prediction_id']} not found")
                else:
                    # Match by (zone_id, hour_ts) — most recent prediction
                    row = self._db.execute(
                        "SELECT prediction_id FROM prediction_log "
                        "WHERE zone_id = ? AND hour_ts = ? "
                        "ORDER BY created_at DESC LIMIT 1",
                        (actual["zone_id"], actual["hour_ts"]),
                    ).fetchone()
                    if row:
                        self._db.execute(
                            "UPDATE prediction_log SET actual_value = ? WHERE prediction_id = ?",
                            (actual["actual_value"], row[0]),
                        )
                        matched += 1
                    else:
                        unmatched += 1
                        warnings.append(
                            f"No prediction found for zone_id={actual['zone_id']}, "
                            f"hour_ts={actual['hour_ts']}"
                        )
            self._db.commit()

        return {
            "matched_count": matched,
            "unmatched_count": unmatched,
            "warnings": warnings,
        }

    def compute_quality(self, window: str = "7d") -> dict:
        """Compute quality metrics over matched pairs in the given window.

        Window format: "<int>d" (e.g. "7d", "30d"). Raises ValueError for
        invalid format or non-positive values.
        """
        stripped = window.rstrip("d")
        if not window.endswith("d") or not stripped.isdigit() or int(stripped) <= 0:
            raise ValueError(f"Invalid window format '{window}', expected '<int>d' (e.g. '7d')")
        days = int(stripped)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = self._db.execute(
            "SELECT predicted_value, actual_value FROM prediction_log "
            "WHERE actual_value IS NOT NULL AND created_at >= ?",
            (cutoff,),
        ).fetchall()

        if len(rows) < 10:
            return {
                "status": "insufficient_matched_pairs",
                "matched": len(rows),
                "required": 10,
            }

        preds = np.array([r[0] for r in rows])
        actuals = np.array([r[1] for r in rows])

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        # sMAPE: bounded [0, 200], handles zeros
        denominator = np.abs(preds) + np.abs(actuals)
        nonzero_mask = denominator > 0
        zero_count = int(np.sum(~nonzero_mask))
        if nonzero_mask.any():
            smape = float(
                np.mean(
                    2
                    * np.abs(preds[nonzero_mask] - actuals[nonzero_mask])
                    / denominator[nonzero_mask]
                )
                * 100
            )
        else:
            smape = 0.0

        return {
            "status": "ok",
            "matched_pairs": len(rows),
            "window": window,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "smape": round(smape, 6),
            "zero_denominator_pairs_excluded": zero_count,
        }
