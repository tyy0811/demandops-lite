"""LightGBM demand prediction model with non-negative clipping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from demandops.models.registry import DemandModel, register_model


@register_model("lightgbm")
class LightGBMModel(DemandModel):
    """LightGBM regressor with predictions clipped to zero.

    predict() returns np.clip(raw_predictions, 0.0, None).
    predict_raw() returns unclipped predictions for tracking clip stats.
    save()/load() use joblib for reliable serialization (fix #4).
    """

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        num_threads: int = -1,
        verbose: int = -1,
        **kwargs: Any,
    ) -> None:
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": num_threads,
            "verbose": verbose,
        }
        self._early_stopping_rounds = early_stopping_rounds
        self._model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = lgb.LGBMRegressor(**self._params)

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["callbacks"] = [
                lgb.early_stopping(self._early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self._model.fit(X, y, **fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with non-negative clipping."""
        return np.clip(self.predict_raw(X), 0.0, None)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict without clipping. Use for tracking clip statistics."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        """Save model via joblib (reliable round-trip, fix #4)."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        """Load model from joblib file."""
        self._model = joblib.load(path)

    def get_params(self) -> dict:
        params = dict(self._params)
        if self._model is not None and hasattr(self._model, "best_iteration_"):
            params["best_iteration"] = self._model.best_iteration_
        return params

    @property
    def feature_importances(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return self._model.feature_importances_

    @property
    def booster(self) -> lgb.LGBMRegressor | None:
        return self._model
