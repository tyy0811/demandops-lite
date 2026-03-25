"""Abstract base class for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class DatasetAdapter(ABC):
    """Adapter interface for dataset-specific download and preparation.

    Each adapter produces a dense hourly history DataFrame with a common
    schema: (zone_id, zone_name, hour_ts, trip_count). All downstream
    pipeline code (feature engineering, training, serving) is dataset-agnostic.
    """

    name: str

    @abstractmethod
    def download(self, raw_dir: Path, months: list[str]) -> list[Path]:
        """Download raw data files. Idempotent -- skips existing files."""

    @abstractmethod
    def prepare_hourly_history(
        self,
        raw_dir: Path,
        processed_dir: Path,
        config: dict,
    ) -> tuple[pl.DataFrame, list[int]]:
        """Raw data -> dense hourly grid.

        Returns:
            Tuple of (history_df, entity_ids) where:
            - history_df has columns: zone_id (int), zone_name (str),
              hour_ts (datetime), trip_count (int), avg_fare (float|null),
              avg_distance (float|null)
            - entity_ids is a sorted list of all entity IDs in the dataset
        """
