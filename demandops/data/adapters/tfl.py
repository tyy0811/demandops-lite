"""TfL Santander Cycle Hire dataset adapter.

Data source: cycling.data.tfl.gov.uk/usage-stats/
Weekly CSVs, no API key. Date format: DD/MM/YYYY HH:MM (UK format).
Some rows include seconds — truncate to str[:16] before parsing.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import polars as pl
import structlog

from demandops.data.adapters.base import DatasetAdapter

logger = structlog.get_logger()

TFL_BASE_URL = "https://cycling.data.tfl.gov.uk/usage-stats"

# Hardcoded filenames for Dec 2023 - Feb 2024.
# No directory scraping — fragile and unnecessary for a fixed date range.
TFL_FILES: dict[str, list[str]] = {
    "2023-12": [
        "393JourneyDataExtract29Nov2023-05Dec2023.csv",
        "394JourneyDataExtract06Dec2023-12Dec2023.csv",
        "395JourneyDataExtract13Dec2023-19Dec2023.csv",
        "396JourneyDataExtract20Dec2023-26Dec2023.csv",
        "397JourneyDataExtract27Dec2023-02Jan2024.csv",
    ],
    "2024-01": [
        "398JourneyDataExtract03Jan2024-09Jan2024.csv",
        "399JourneyDataExtract10Jan2024-16Jan2024.csv",
        "400JourneyDataExtract17Jan2024-23Jan2024.csv",
        "401JourneyDataExtract24Jan2024-30Jan2024.csv",
    ],
    "2024-02": [
        "402JourneyDataExtract31Jan2024-06Feb2024.csv",
        "403JourneyDataExtract07Feb2024-13Feb2024.csv",
        "404JourneyDataExtract14Feb2024-20Feb2024.csv",
        "405JourneyDataExtract21Feb2024-27Feb2024.csv",
        "406JourneyDataExtract28Feb2024-05Mar2024.csv",
    ],
}


def _atomic_download(url: str, dest: Path) -> None:
    """Download to a temp file, then atomically rename on success."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urlretrieve(url, tmp)
        tmp.rename(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class TfLAdapter(DatasetAdapter):
    """TfL Santander Cycle Hire: weekly CSVs from cycling.data.tfl.gov.uk."""

    name = "tfl"

    def download(self, raw_dir: Path, months: list[str]) -> list[Path]:
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for month in months:
            files = TFL_FILES.get(month, [])
            if not files:
                logger.warning("no_tfl_files_for_month", month=month)
                continue
            for filename in files:
                dest = raw_dir / filename
                if dest.exists():
                    logger.info("file_exists_skipping", path=str(dest))
                else:
                    url = f"{TFL_BASE_URL}/{filename}"
                    logger.info("downloading", url=url, dest=str(dest))
                    _atomic_download(url, dest)
                    logger.info("download_complete", path=str(dest))
                paths.append(dest)
        return paths

    def _parse_csv(self, csv_path: Path) -> pl.DataFrame:
        """Parse a single TfL CSV into a trips DataFrame.

        Handles UK date format (DD/MM/YYYY HH:MM) and truncates to hour.
        """
        df = pl.read_csv(csv_path, try_parse_dates=False)

        # Rename to common schema
        df = df.rename(
            {
                "StartStation Id": "zone_id",
                "StartStation Name": "zone_name",
            }
        )

        # Parse UK date: truncate to first 16 chars (inconsistent seconds)
        df = df.with_columns(
            pl.col("Start Date")
            .str.slice(0, 16)
            .str.to_datetime("%d/%m/%Y %H:%M")
            .dt.truncate("1h")
            .alias("hour_ts")
        )

        # Cast zone_id to int (may be string in some files)
        df = df.with_columns(pl.col("zone_id").cast(pl.Int64))

        # Drop rows with null station IDs
        df = df.drop_nulls(subset=["zone_id", "hour_ts"])

        return df.select(["zone_id", "zone_name", "hour_ts"])

    def _aggregate_hourly(self, trips: pl.DataFrame) -> pl.DataFrame:
        """Aggregate trips to hourly counts per station."""
        return (
            trips.group_by(["zone_id", "zone_name", "hour_ts"])
            .agg(pl.len().alias("trip_count"))
            .sort(["zone_id", "hour_ts"])
        )

    def prepare_hourly_history(
        self,
        raw_dir: Path,
        processed_dir: Path,
        config: dict,
    ) -> tuple[pl.DataFrame, list[int]]:
        months = config["data"]["months"]

        # Parse and concatenate all CSV files — every file must be present.
        # A missing weekly extract would silently become zero-demand rows after
        # densification, corrupting the benchmark dataset.
        all_trips: list[pl.DataFrame] = []
        for month in months:
            files = TFL_FILES.get(month, [])
            for filename in files:
                csv_path = raw_dir / filename
                if not csv_path.exists():
                    raise FileNotFoundError(
                        f"Missing TfL CSV: {csv_path}. Run download first. "
                        f"A missing file would corrupt the dense grid with synthetic zeros."
                    )
                trips = self._parse_csv(csv_path)
                all_trips.append(trips)

        if not all_trips:
            raise FileNotFoundError("No TfL CSV files found. Run download first.")

        trips_df = pl.concat(all_trips)
        logger.info("tfl_trips_loaded", total_trips=len(trips_df))

        # Aggregate to hourly
        hourly = self._aggregate_hourly(trips_df)

        # Determine station universe
        zone_ids = sorted(hourly["zone_id"].unique().to_list())
        logger.info("tfl_stations_found", n_stations=len(zone_ids))

        # Build zone_name lookup from first occurrence
        zone_name_lookup = trips_df.group_by("zone_id").agg(pl.col("zone_name").first())

        # Densify grid using DuckDB
        con = duckdb.connect()
        con.register("hourly_agg", hourly.to_pandas())

        zone_ids_sql = ", ".join(str(z) for z in zone_ids)
        con.execute(f"""
            CREATE OR REPLACE TABLE dense_grid AS
            WITH zones AS (
                SELECT unnest([{zone_ids_sql}]) AS zone_id
            ),
            hours AS (
                SELECT unnest(
                    generate_series(
                        TIMESTAMP '2023-12-01 00:00:00',
                        TIMESTAMP '2024-02-29 23:00:00',
                        INTERVAL '1 hour'
                    )
                ) AS hour_ts
            ),
            grid AS (
                SELECT z.zone_id, h.hour_ts
                FROM zones z
                CROSS JOIN hours h
            )
            SELECT
                g.zone_id,
                g.hour_ts,
                COALESCE(a.trip_count, 0)::INTEGER AS trip_count
            FROM grid g
            LEFT JOIN hourly_agg a
                ON g.zone_id = a.zone_id AND g.hour_ts = a.hour_ts
        """)

        dense_df = con.execute("SELECT * FROM dense_grid").pl()
        con.close()

        # Join zone names
        dense_df = dense_df.join(zone_name_lookup, on="zone_id", how="left")
        dense_df = dense_df.with_columns(
            pl.col("zone_name").fill_null(pl.lit("Unknown Station")),
            pl.lit(None).cast(pl.Float64).alias("avg_fare"),
            pl.lit(None).cast(pl.Float64).alias("avg_distance"),
        )

        # Reorder to match HourlyHistorySchema
        dense_df = dense_df.select(
            [
                "zone_id",
                "zone_name",
                "hour_ts",
                "trip_count",
                "avg_fare",
                "avg_distance",
            ]
        )

        logger.info("tfl_history_prepared", rows=len(dense_df), stations=len(zone_ids))
        return dense_df, zone_ids
