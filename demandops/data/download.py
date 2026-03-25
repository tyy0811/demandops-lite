"""Download raw data using the appropriate dataset adapter."""

from __future__ import annotations

from pathlib import Path

import structlog

from demandops.data.adapters.base import DatasetAdapter

logger = structlog.get_logger()


def download_all(
    adapter: DatasetAdapter,
    months: list[str],
    raw_dir: Path,
    zones_path: Path | None = None,
) -> dict:
    """Download all data using the adapter. Returns dict with paths."""
    month_paths = adapter.download(raw_dir, months)
    result: dict = {"months": month_paths}

    if hasattr(adapter, "download_zones") and zones_path is not None:
        zone_path = adapter.download_zones(zones_path)
        result["zones"] = zone_path

    return result
