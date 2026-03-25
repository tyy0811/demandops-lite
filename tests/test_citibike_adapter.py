"""Tests for Citibike NYC adapter using mock CSV data."""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


CITIBIKE_CSV_HEADER = (
    "ride_id,rideable_type,started_at,ended_at,"
    "start_station_name,start_station_id,"
    "end_station_name,end_station_id,"
    "start_lat,start_lng,end_lat,end_lng,member_casual\n"
)


def _make_row(
    ride_id: str,
    started_at: str,
    station_name: str,
    station_id: str,
    ended_at: str = "2024-01-01 10:30:00",
) -> str:
    return (
        f"{ride_id},classic_bike,{started_at},{ended_at},"
        f"{station_name},{station_id},"
        f"End Station,9999,"
        f"40.7,-74.0,40.8,-73.9,member\n"
    )


@pytest.fixture
def mock_citibike_csv(tmp_path: Path) -> Path:
    """Create a mock Citibike CSV with new-schema columns."""
    content = CITIBIKE_CSV_HEADER
    content += _make_row("R1", "2024-01-01 10:15:00", "Station Alpha", "1000.03")
    content += _make_row("R2", "2024-01-01 10:45:00", "Station Alpha", "1000.03")
    content += _make_row("R3", "2024-01-01 11:00:00", "Station Beta", "2000.01")
    content += _make_row("R4", "2024-01-01 11:30:00", "Station Alpha", "1000.03")
    content += _make_row("R5", "2024-01-01 11:05:00", "Station Beta", "2000.01")

    csv_path = tmp_path / "202401-citibike-tripdata.csv"
    csv_path.write_text(content)
    return csv_path


@pytest.fixture
def mock_citibike_csv_with_bad_ids(tmp_path: Path) -> Path:
    """CSV with a mix of valid decimal IDs and non-numeric station IDs."""
    content = CITIBIKE_CSV_HEADER
    content += _make_row("R1", "2024-01-01 10:00:00", "Good Station", "5678.02")
    content += _make_row("R2", "2024-01-01 10:05:00", "Letter ID", "AB123")
    content += _make_row("R3", "2024-01-01 10:10:00", "System ID", "SYS016")
    content += _make_row("R4", "2024-01-01 10:15:00", "Also Good", "9999.01")
    # Row with empty station ID
    content += (
        "R5,classic_bike,2024-01-01 10:20:00,2024-01-01 10:50:00,"
        "No ID Station,,"
        "End Station,9999.01,"
        "40.7,-74.0,40.8,-73.9,member\n"
    )

    csv_path = tmp_path / "202401-citibike-tripdata-bad.csv"
    csv_path.write_text(content)
    return csv_path


class TestCitibikeAdapter:
    def test_parse_citibike_csv(self, mock_citibike_csv) -> None:
        """Citibike CSV parsed correctly into trips DataFrame."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        adapter = CitibikeAdapter()
        trips, dropped = adapter._parse_csv(mock_citibike_csv)

        assert len(trips) == 5
        assert dropped == 0
        assert set(trips.columns) == {"zone_id", "zone_name", "hour_ts"}

    def test_datetime_truncated_to_hour(self, mock_citibike_csv) -> None:
        """started_at truncated to hour (minutes/seconds stripped)."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        adapter = CitibikeAdapter()
        trips, _ = adapter._parse_csv(mock_citibike_csv)

        hours = sorted(trips["hour_ts"].to_list())
        # 10:15 and 10:45 -> hour 10; 11:00, 11:05, 11:30 -> hour 11
        assert hours[0] == datetime(2024, 1, 1, 10, 0)
        assert hours[-1] == datetime(2024, 1, 1, 11, 0)

    def test_station_id_mapped_to_zone_id(self, mock_citibike_csv) -> None:
        """start_station_id mapped to zone_id as int."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        adapter = CitibikeAdapter()
        trips, _ = adapter._parse_csv(mock_citibike_csv)

        # "1000.03" -> 100003, "2000.01" -> 200001
        zone_ids = set(trips["zone_id"].to_list())
        assert zone_ids == {100003, 200001}

    def test_filter_non_numeric_station_ids(
        self, mock_citibike_csv_with_bad_ids, capsys
    ) -> None:
        """Rows with non-numeric station IDs are dropped and logged."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        adapter = CitibikeAdapter()
        trips, dropped = adapter._parse_csv(mock_citibike_csv_with_bad_ids)

        # 5 rows total: "5678.02" kept, "AB123" dropped, "SYS016" dropped,
        # "9999.01" kept, empty dropped = 3 dropped
        assert len(trips) == 2
        assert dropped == 3
        assert set(trips["zone_id"].to_list()) == {567802, 999901}

        # Verify filtering is logged (structlog writes to stdout)
        captured = capsys.readouterr()
        assert "citibike_filtered_non_numeric" in captured.out
        assert "dropped=3" in captured.out

    def test_aggregation_counts_trips(self, mock_citibike_csv) -> None:
        """Hourly aggregation counts trips per station per hour."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        adapter = CitibikeAdapter()
        trips, _ = adapter._parse_csv(mock_citibike_csv)
        hourly = adapter._aggregate_hourly(trips)

        # Station 100003 (1000.03), hour 10: 2 trips (R1 10:15, R2 10:45)
        s1000_h10 = hourly.filter(
            (pl.col("zone_id") == 100003)
            & (pl.col("hour_ts") == datetime(2024, 1, 1, 10))
        )
        assert s1000_h10["trip_count"][0] == 2

        # Station 200001 (2000.01), hour 11: 2 trips (R3 11:00, R5 11:05)
        s2000_h11 = hourly.filter(
            (pl.col("zone_id") == 200001)
            & (pl.col("hour_ts") == datetime(2024, 1, 1, 11))
        )
        assert s2000_h11["trip_count"][0] == 2

    def test_prepare_hourly_history_end_to_end(self, tmp_path) -> None:
        """Full prepare_hourly_history: dense grid, schema, zero-fill."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        # Create mock CSVs for Jan 2024 (two files, simulating multi-CSV zip)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        csv1 = CITIBIKE_CSV_HEADER
        csv1 += _make_row("R1", "2024-01-03 10:00:00", "Alpha", "500.01")
        csv1 += _make_row("R2", "2024-01-03 10:05:00", "Beta", "501.02")
        (raw_dir / "202401-citibike-tripdata_1.csv").write_text(csv1)

        csv2 = CITIBIKE_CSV_HEADER
        csv2 += _make_row("R3", "2024-01-10 14:00:00", "Alpha", "500.01")
        (raw_dir / "202401-citibike-tripdata_2.csv").write_text(csv2)

        config = {"data": {"months": ["2024-01"]}}
        adapter = CitibikeAdapter()
        history_df, zone_ids = adapter.prepare_hourly_history(
            raw_dir, tmp_path / "processed", config
        )

        # "500.01" -> 50001, "501.02" -> 50102
        assert sorted(zone_ids) == [50001, 50102]

        # Verify schema columns
        assert set(history_df.columns) == {
            "zone_id", "zone_name", "hour_ts", "trip_count",
            "avg_fare", "avg_distance",
        }

        # Verify dense grid: 2 stations x 2184 hours (Jan 1 00:00 - Mar 31 23:00)
        assert len(history_df) == 2 * 2184

        # Verify non-zero counts
        s500_h10 = history_df.filter(
            (pl.col("zone_id") == 50001)
            & (pl.col("hour_ts") == datetime(2024, 1, 3, 10))
        )
        assert s500_h10["trip_count"][0] == 1

        # Verify zero-fill: station 50102 at hour 14 on Jan 10 had no trips
        s501_h14 = history_df.filter(
            (pl.col("zone_id") == 50102)
            & (pl.col("hour_ts") == datetime(2024, 1, 10, 14))
        )
        assert s501_h14["trip_count"][0] == 0

        # Verify avg_fare and avg_distance are null (Citibike has no fare data)
        assert history_df["avg_fare"].null_count() == len(history_df)
        assert history_df["avg_distance"].null_count() == len(history_df)

        # Validate against HourlyHistorySchema
        from demandops.data.schemas import HourlyHistorySchema

        HourlyHistorySchema.validate(history_df)

    def test_multi_csv_zip_extraction(self, tmp_path) -> None:
        """Multiple CSVs in a zip are all extracted."""
        from demandops.data.adapters.citibike import _extract_csvs

        csv1 = CITIBIKE_CSV_HEADER + _make_row(
            "R1", "2024-01-01 10:00:00", "Alpha", "100.01"
        )
        csv2 = CITIBIKE_CSV_HEADER + _make_row(
            "R2", "2024-01-15 12:00:00", "Beta", "200.02"
        )

        zip_path = tmp_path / "202401-citibike-tripdata.csv.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("202401-citibike-tripdata_1.csv", csv1)
            zf.writestr("202401-citibike-tripdata_2.csv", csv2)

        extracted = _extract_csvs(zip_path, tmp_path)

        assert len(extracted) == 2
        assert all(p.exists() for p in extracted)
        assert {p.name for p in extracted} == {
            "202401-citibike-tripdata_1.csv",
            "202401-citibike-tripdata_2.csv",
        }

    def test_zip_extraction_idempotent(self, tmp_path) -> None:
        """Re-extracting a zip does not overwrite existing CSVs."""
        from demandops.data.adapters.citibike import _extract_csvs

        csv_content = CITIBIKE_CSV_HEADER + _make_row(
            "R1", "2024-01-01 10:00:00", "Alpha", "100.01"
        )

        zip_path = tmp_path / "202401-citibike-tripdata.csv.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("202401-citibike-tripdata.csv", csv_content)

        # First extraction
        _extract_csvs(zip_path, tmp_path)
        csv_path = tmp_path / "202401-citibike-tripdata.csv"
        mtime_first = csv_path.stat().st_mtime

        # Second extraction — file should not be overwritten
        _extract_csvs(zip_path, tmp_path)
        assert csv_path.stat().st_mtime == mtime_first

    def test_missing_csv_raises_error(self, tmp_path) -> None:
        """Missing CSV files raise FileNotFoundError, not silent zeros."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        adapter = CitibikeAdapter()
        config = {"data": {"months": ["2024-01"]}}
        with pytest.raises(FileNotFoundError, match="No Citibike CSVs found"):
            adapter.prepare_hourly_history(tmp_path, tmp_path, config)

    def test_single_digit_decimal_normalized(self, tmp_path) -> None:
        """Single-digit decimals like '3113.1' are normalized to '3113.10' -> 311310."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        content = CITIBIKE_CSV_HEADER
        content += _make_row("R1", "2024-01-01 10:00:00", "Station A", "3113.1")
        content += _make_row("R2", "2024-01-01 10:05:00", "Station A", "3113.10")

        csv_path = tmp_path / "202401-citibike-tripdata.csv"
        csv_path.write_text(content)

        adapter = CitibikeAdapter()
        trips, _ = adapter._parse_csv(csv_path)

        # Both should map to the same zone_id
        assert set(trips["zone_id"].to_list()) == {311310}

    def test_all_rows_filtered_raises_error(self, tmp_path) -> None:
        """ValueError raised when all rows have non-numeric station IDs."""
        from demandops.data.adapters.citibike import CitibikeAdapter

        content = CITIBIKE_CSV_HEADER
        content += _make_row("R1", "2024-01-01 10:00:00", "Bad A", "SYS016")
        content += _make_row("R2", "2024-01-01 11:00:00", "Bad B", "JC009")

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "202401-citibike-tripdata.csv").write_text(content)

        adapter = CitibikeAdapter()
        config = {"data": {"months": ["2024-01"]}}
        with pytest.raises(ValueError, match="All .* rows were filtered"):
            adapter.prepare_hourly_history(raw_dir, tmp_path / "processed", config)

    def test_download_fetches_and_extracts(self, tmp_path) -> None:
        """download() fetches zip from network and extracts CSVs."""
        from unittest.mock import patch

        from demandops.data.adapters import citibike as citibike_module
        from demandops.data.adapters.citibike import CitibikeAdapter

        # Create a mock zip that _atomic_download will "produce"
        csv_content = CITIBIKE_CSV_HEADER + _make_row(
            "R1", "2024-01-01 10:00:00", "Alpha", "100.01"
        )
        zip_path = tmp_path / "staged.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("202401-citibike-tripdata.csv", csv_content)
        zip_bytes = zip_path.read_bytes()

        raw_dir = tmp_path / "raw"

        def fake_download(url: str, dest: Path) -> None:
            dest.write_bytes(zip_bytes)

        adapter = CitibikeAdapter()
        with patch.object(citibike_module, "CITIBIKE_FILES", {"2024-01": "202401-citibike-tripdata.csv.zip"}), \
             patch.object(citibike_module, "_atomic_download", side_effect=fake_download):
            paths = adapter.download(raw_dir, ["2024-01"])

        # Verify zip was downloaded and CSV extracted
        assert len(paths) == 1
        assert paths[0].name == "202401-citibike-tripdata.csv"
        assert paths[0].exists()

    def test_download_skips_when_csvs_exist(self, tmp_path) -> None:
        """download() skips network fetch when extracted CSVs already exist."""
        from unittest.mock import patch

        from demandops.data.adapters import citibike as citibike_module
        from demandops.data.adapters.citibike import CitibikeAdapter

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # Pre-existing extracted CSV
        existing = raw_dir / "202401-citibike-tripdata.csv"
        existing.write_text(CITIBIKE_CSV_HEADER)

        adapter = CitibikeAdapter()
        with patch.object(citibike_module, "CITIBIKE_FILES", {"2024-01": "202401-citibike-tripdata.csv.zip"}), \
             patch.object(citibike_module, "_atomic_download") as mock_dl:
            paths = adapter.download(raw_dir, ["2024-01"])

        # Network should never be called
        mock_dl.assert_not_called()
        assert len(paths) == 1
        assert paths[0] == existing
