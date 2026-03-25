"""Dataset adapters for multi-dataset pipeline support."""

from demandops.data.adapters.base import DatasetAdapter
from demandops.data.adapters.taxi import TaxiAdapter
from demandops.data.adapters.tfl import TfLAdapter

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "taxi": TaxiAdapter,
    "tfl": TfLAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    """Get adapter instance by name."""
    if name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[name]()


__all__ = ["DatasetAdapter", "TaxiAdapter", "TfLAdapter", "get_adapter"]
