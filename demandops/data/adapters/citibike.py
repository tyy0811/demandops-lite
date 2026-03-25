"""Citibike NYC dataset adapter.

Data source: s3.amazonaws.com/tripdata/
Monthly zip files containing CSVs. Date format: ISO (YYYY-MM-DD HH:MM:SS).
Station IDs are strings — non-numeric IDs are filtered (see DECISIONS.md #21).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import polars as pl
import structlog

from demandops.data.adapters.base import DatasetAdapter

logger = structlog.get_logger()

CITIBIKE_BASE_URL = "https://s3.amazonaws.com/tripdata"

CITIBIKE_FILES: dict[str, str] = {
    "2024-01": "202401-citibike-tripdata.zip",
    "2024-02": "202402-citibike-tripdata.zip",
    "2024-03": "202403-citibike-tripdata.zip",
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


def _extract_csvs(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract CSV files from a zip, flattening any directory structure."""
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            dest = dest_dir / Path(name).name
            if not dest.exists():
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                logger.info("extracted_csv", zip=zip_path.name, csv=dest.name)
            extracted.append(dest)
    return extracted


class CitibikeAdapter(DatasetAdapter):
    """Citibike NYC: monthly zipped CSVs from s3.amazonaws.com/tripdata."""

    name = "citibike"

    def download(self, raw_dir: Path, months: list[str]) -> list[Path]:
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for month in months:
            zip_filename = CITIBIKE_FILES.get(month)
            if not zip_filename:
                logger.warning("no_citibike_files_for_month", month=month)
                continue

            # Check if CSVs are already extracted — skip download entirely
            # if so (re-extraction is cheap, re-download is slow).
            month_compact = month.replace("-", "")
            existing_csvs = sorted(raw_dir.glob(f"{month_compact}*citibike*.csv"))
            if existing_csvs:
                logger.info(
                    "csvs_exist_skipping",
                    month=month,
                    n_files=len(existing_csvs),
                )
                paths.extend(existing_csvs)
                continue

            zip_path = raw_dir / zip_filename
            if not zip_path.exists():
                url = f"{CITIBIKE_BASE_URL}/{zip_filename}"
                logger.info("downloading", url=url, dest=str(zip_path))
                _atomic_download(url, zip_path)
                logger.info("download_complete", path=str(zip_path))
            extracted = _extract_csvs(zip_path, raw_dir)
            paths.extend(extracted)
        return paths

    def _parse_csv(self, csv_path: Path) -> tuple[pl.DataFrame, int]:
        """Parse a single Citibike CSV into a trips DataFrame.

        Returns (trips_df, dropped_count) where dropped_count is the number
        of rows filtered due to null or non-numeric station IDs.
        """
        df = pl.read_csv(
            csv_path,
            try_parse_dates=False,
            schema_overrides={
                "start_station_id": pl.Utf8,
                "end_station_id": pl.Utf8,
            },
        )
        total_rows = len(df)

        df = df.rename({
            "start_station_id": "zone_id",
            "start_station_name": "zone_name",
        })

        # Station IDs are decimal strings like "4028.03" (NNNN.DD format).
        # Some have a single decimal digit ("3113.1" == "3113.10").
        # A handful are system IDs ("SYS016", "JC009") — drop those.
        # Convert to int by normalizing to 2 decimal places and removing
        # the dot: "4028.03" -> 402803, "3113.1" -> 311310.
        df = df.filter(
            pl.col("zone_id").is_not_null()
            & pl.col("zone_id").cast(pl.Utf8).str.contains(r"^\d+\.\d{1,2}$")
        )
        dropped = total_rows - len(df)

        if dropped > 0:
            logger.info(
                "citibike_filtered_non_numeric",
                file=csv_path.name,
                dropped=dropped,
                kept=len(df),
            )

        # Normalize: pad single-digit decimals to 2 digits, remove dot, cast to int
        # "4028.03" -> "402803", "3113.1" -> "3113.10" -> "311310"
        df = df.with_columns(
            pl.col("zone_id").cast(pl.Utf8)
            .str.replace(r"\.(\d)$", ".${1}0")  # pad "3113.1" -> "3113.10"
            .str.replace(r"\.", "")            # remove dot
            .cast(pl.Int64)
        )

        # Parse started_at: ISO format, some rows have milliseconds.
        # Truncate string to 19 chars ("YYYY-MM-DD HH:MM:SS") before parsing.
        df = df.with_columns(
            pl.col("started_at")
            .str.slice(0, 19)
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .dt.truncate("1h")
            .alias("hour_ts")
        )

        return df.select(["zone_id", "zone_name", "hour_ts"]), dropped

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

        all_trips: list[pl.DataFrame] = []
        total_dropped = 0
        for month in months:
            month_compact = month.replace("-", "")  # "2024-01" -> "202401"
            csvs = sorted(raw_dir.glob(f"{month_compact}*citibike*.csv"))
            if not csvs:
                raise FileNotFoundError(
                    f"No Citibike CSVs found for {month} in {raw_dir}. "
                    f"Run download first. Missing files would corrupt the "
                    f"dense grid with synthetic zeros."
                )
            for csv_path in csvs:
                trips, dropped = self._parse_csv(csv_path)
                total_dropped += dropped
                all_trips.append(trips)

        if not all_trips:
            raise FileNotFoundError("No Citibike CSV files found. Run download first.")

        trips_df = pl.concat(all_trips)
        if trips_df.is_empty():
            raise ValueError(
                f"All {total_dropped} rows were filtered (non-numeric station IDs). "
                f"No valid trips remain — cannot build dense grid."
            )
        logger.info(
            "citibike_trips_loaded",
            total_trips=len(trips_df),
            rows_filtered=total_dropped,
        )

        hourly = self._aggregate_hourly(trips_df)

        zone_ids = sorted(hourly["zone_id"].unique().to_list())
        logger.info("citibike_stations_found", n_stations=len(zone_ids))

        zone_name_lookup = trips_df.group_by("zone_id").agg(
            pl.col("zone_name").first()
        )

        # Densify with DuckDB cross-join
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
                        TIMESTAMP '2024-01-01 00:00:00',
                        TIMESTAMP '2024-03-31 23:00:00',
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

        dense_df = dense_df.join(zone_name_lookup, on="zone_id", how="left")
        dense_df = dense_df.with_columns(
            pl.col("zone_name").fill_null(pl.lit("Unknown Station")),
            pl.lit(None).cast(pl.Float64).alias("avg_fare"),
            pl.lit(None).cast(pl.Float64).alias("avg_distance"),
        )

        dense_df = dense_df.select([
            "zone_id", "zone_name", "hour_ts", "trip_count",
            "avg_fare", "avg_distance",
        ])

        logger.info(
            "citibike_history_prepared", rows=len(dense_df), stations=len(zone_ids)
        )
        return dense_df, zone_ids
