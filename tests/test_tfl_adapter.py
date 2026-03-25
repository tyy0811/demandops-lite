"""Tests for TfL Cycle Hire adapter using mock CSV data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def mock_tfl_csv(tmp_path: Path) -> Path:
    """Create a mock TfL CSV file with known data."""
    csv_content = (
        "Rental Id,Duration,Bike Id,End Date,EndStation Id,EndStation Name,"
        "Start Date,StartStation Id,StartStation Name\n"
        "1,600,123,01/01/2024 10:10,100,Station A End,01/01/2024 10:00,200,Station Alpha\n"
        "2,300,456,01/01/2024 10:05,101,Station B End,01/01/2024 10:01,200,Station Alpha\n"
        "3,900,789,01/01/2024 11:15,100,Station A End,01/01/2024 11:00,201,Station Beta\n"
        "4,450,012,01/01/2024 11:30,101,Station B End,01/01/2024 11:05,200,Station Alpha\n"
        "5,500,345,01/01/2024 12:00,100,Station A End,01/01/2024 11:30,201,Station Beta\n"
    )
    csv_path = tmp_path / "test_tfl_data.csv"
    csv_path.write_text(csv_content)
    return csv_path


class TestTfLAdapter:
    def test_parse_tfl_csv(self, mock_tfl_csv) -> None:
        """TfL CSV parsed correctly into trips DataFrame."""
        from demandops.data.adapters.tfl import TfLAdapter

        adapter = TfLAdapter()
        trips = adapter._parse_csv(mock_tfl_csv)

        assert len(trips) == 5
        assert "zone_id" in trips.columns
        assert "zone_name" in trips.columns
        assert "hour_ts" in trips.columns

    def test_datetime_truncated_to_hour(self, mock_tfl_csv) -> None:
        """Start Date truncated to hour (minutes stripped)."""
        from demandops.data.adapters.tfl import TfLAdapter

        adapter = TfLAdapter()
        trips = adapter._parse_csv(mock_tfl_csv)

        # All trips on 01/01/2024 — hours should be 10, 10, 11, 11, 11
        hours = sorted(trips["hour_ts"].to_list())
        assert hours[0] == datetime(2024, 1, 1, 10, 0)
        assert hours[-1] == datetime(2024, 1, 1, 11, 0)

    def test_station_id_mapped_to_zone_id(self, mock_tfl_csv) -> None:
        """StartStation Id mapped to zone_id."""
        from demandops.data.adapters.tfl import TfLAdapter

        adapter = TfLAdapter()
        trips = adapter._parse_csv(mock_tfl_csv)

        zone_ids = set(trips["zone_id"].to_list())
        assert zone_ids == {200, 201}

    def test_aggregation_counts_trips(self, mock_tfl_csv) -> None:
        """Hourly aggregation counts trips per station per hour."""
        from demandops.data.adapters.tfl import TfLAdapter

        adapter = TfLAdapter()
        trips = adapter._parse_csv(mock_tfl_csv)
        hourly = adapter._aggregate_hourly(trips)

        # Station 200, hour 10: 2 trips
        s200_h10 = hourly.filter(
            (pl.col("zone_id") == 200) & (pl.col("hour_ts") == datetime(2024, 1, 1, 10))
        )
        assert s200_h10["trip_count"][0] == 2

        # Station 201, hour 11: 2 trips
        s201_h11 = hourly.filter(
            (pl.col("zone_id") == 201) & (pl.col("hour_ts") == datetime(2024, 1, 1, 11))
        )
        assert s201_h11["trip_count"][0] == 2
