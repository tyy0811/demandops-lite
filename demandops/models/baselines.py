"""Baseline models: Historical Slot Mean and Seasonal Naive."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from demandops.features import IDX_DAY_OF_WEEK, IDX_HOUR_OF_DAY, IDX_LAG_168H, IDX_ZONE_ID
from demandops.models.registry import DemandModel, register_model


@register_model("slot_mean")
class HistoricalSlotMean(DemandModel):
    """Predict the historical mean for each (zone_id, hour_of_day, day_of_week) slot.

    Keyed per zone so that zones with different demand levels produce
    different predictions at the same time slot. Predictions are naturally
    non-negative (mean of non-negative counts).
    """

    name = "slot_mean"

    def __init__(self, **kwargs: Any) -> None:
        self._slot_means: dict[tuple[int, int, int], float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        slot_sums: dict[tuple[int, int, int], float] = defaultdict(float)
        slot_counts: dict[tuple[int, int, int], int] = defaultdict(int)

        for i in range(len(X)):
            key = (
                int(X[i, IDX_ZONE_ID]),
                int(X[i, IDX_HOUR_OF_DAY]),
                int(X[i, IDX_DAY_OF_WEEK]),
            )
            slot_sums[key] += y[i]
            slot_counts[key] += 1

        self._slot_means = {k: slot_sums[k] / slot_counts[k] for k in slot_sums}
        self._global_mean = float(np.mean(y)) if len(y) > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.empty(len(X))
        for i in range(len(X)):
            key = (
                int(X[i, IDX_ZONE_ID]),
                int(X[i, IDX_HOUR_OF_DAY]),
                int(X[i, IDX_DAY_OF_WEEK]),
            )
            preds[i] = self._slot_means.get(key, self._global_mean)
        return preds

    def get_params(self) -> dict:
        return {"n_slots": len(self._slot_means)}


@register_model("seasonal_naive")
class SeasonalNaive(DemandModel):
    """Predict using lag_168h (same hour, same day, one week ago).

    Predictions are exact historical counts — non-negative by definition.
    """

    name = "seasonal_naive"

    def __init__(self, **kwargs: Any) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, IDX_LAG_168H].copy()

    def get_params(self) -> dict:
        return {"lag_column": "lag_168h"}
