"""Model interface and factory registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DemandModel(ABC):
    """Base class for all demand prediction models."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Must return non-negative values."""

    def get_params(self) -> dict:
        """Return model parameters for logging."""
        return {}


MODEL_REGISTRY: dict[str, type[DemandModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def wrapper(cls: type[DemandModel]):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def create_model(name: str, **kwargs: Any) -> DemandModel:
    """Factory: create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
