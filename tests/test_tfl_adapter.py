"""Tests for TfL Cycle Hire adapter using mock CSV data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def mock_tfl_csv(tmp_path: Path) -> Path:
    """Create a mock TfL CSV file matching the 2023-2024 schema."""
    csv_content = (
        '"Number","Start date","Start station number","Start station",'
        '"End date","End station number","End station",'
        '"Bike number","Bike model","Total duration","Total duration (ms)"\n'
        '"1","2024-01-01 10:00","200","Station Alpha",'
        '"2024-01-01 10:10","100","Station A End",'
        '"123","CLASSIC","10m 0s","600000"\n'
        '"2","2024-01-01 10:01","200","Station Alpha",'
        '"2024-01-01 10:05","101","Station B End",'
        '"456","CLASSIC","4m 0s","240000"\n'
        '"3","2024-01-01 11:00","201","Station Beta",'
        '"2024-01-01 11:15","100","Station A End",'
        '"789","CLASSIC","15m 0s","900000"\n'
        '"4","2024-01-01 11:05","200","Station Alpha",'
        '"2024-01-01 11:30","101","Station B End",'
        '"12","CLASSIC","25m 0s","1500000"\n'
        '"5","2024-01-01 11:30","201","Station Beta",'
        '"2024-01-01 12:00","100","Station A End",'
        '"345","CLASSIC","30m 0s","1800000"\n'
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

    def test_prepare_hourly_history_end_to_end(self, tmp_path) -> None:
        """Full prepare_hourly_history() with mock CSVs: dense grid, schema, completeness."""
        from unittest.mock import patch

        from demandops.data.adapters import tfl as tfl_module
        from demandops.data.adapters.tfl import TfLAdapter

        # Create two mock CSVs for a single month (Jan 2024, two bi-weekly files)
        header = (
            '"Number","Start date","Start station number","Start station",'
            '"End date","End station number","End station",'
            '"Bike number","Bike model","Total duration","Total duration (ms)"\n'
        )
        week1 = header + (
            '"1","2024-01-03 10:00","500","Alpha",'
            '"2024-01-03 10:10","99","End A",'
            '"10","CLASSIC","10m","600000"\n'
            '"2","2024-01-03 10:05","501","Beta",'
            '"2024-01-03 10:20","99","End A",'
            '"11","CLASSIC","15m","900000"\n'
        )
        week2 = header + (
            '"3","2024-01-10 14:00","500","Alpha",'
            '"2024-01-10 14:30","99","End A",'
            '"12","CLASSIC","30m","1800000"\n'
        )

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "week1.csv").write_text(week1)
        (raw_dir / "week2.csv").write_text(week2)

        # Patch TFL_FILES to use our mock filenames for a single month
        mock_files = {"2024-01": ["week1.csv", "week2.csv"]}

        config = {
            "data": {"months": ["2024-01"]},
        }

        adapter = TfLAdapter()
        with patch.object(tfl_module, "TFL_FILES", mock_files):
            history_df, zone_ids = adapter.prepare_hourly_history(
                raw_dir, tmp_path / "processed", config
            )

        # Verify zone universe
        assert sorted(zone_ids) == [500, 501]

        # Verify schema columns match HourlyHistorySchema
        assert set(history_df.columns) == {
            "zone_id",
            "zone_name",
            "hour_ts",
            "trip_count",
            "avg_fare",
            "avg_distance",
        }

        # Verify dense grid: 2 stations × 744 hours (Jan has 31 days × 24h)
        # Grid covers Dec 1 00:00 through Feb 29 23:00 = 2184 hours per station
        # But with only Jan in months, we still get full Dec-Feb grid from DuckDB
        n_stations = len(zone_ids)
        assert len(history_df) == n_stations * 2184  # 2 stations × 2184 hours

        # Verify non-zero counts exist where we placed trips
        from datetime import datetime

        s500_h10 = history_df.filter(
            (pl.col("zone_id") == 500) & (pl.col("hour_ts") == datetime(2024, 1, 3, 10))
        )
        assert s500_h10["trip_count"][0] == 1  # 1 trip from station 500 at that hour

        # Verify zero-fill: station 501 at hour 14 on Jan 10 had no trips
        s501_h14 = history_df.filter(
            (pl.col("zone_id") == 501) & (pl.col("hour_ts") == datetime(2024, 1, 10, 14))
        )
        assert s501_h14["trip_count"][0] == 0

        # Verify avg_fare and avg_distance are null (TfL has no fare data)
        assert history_df["avg_fare"].null_count() == len(history_df)
        assert history_df["avg_distance"].null_count() == len(history_df)

    def test_missing_csv_raises_error(self, tmp_path) -> None:
        """Missing CSV file raises FileNotFoundError, not silent zeros."""
        from demandops.data.adapters.tfl import TfLAdapter

        adapter = TfLAdapter()
        config = {
            "data": {"months": ["2024-01"]},
        }
        # raw_dir exists but has no CSV files
        with pytest.raises(FileNotFoundError, match="Missing TfL CSV"):
            adapter.prepare_hourly_history(tmp_path, tmp_path, config)
