"""NYC TLC Yellow Taxi dataset adapter."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import polars as pl
import structlog

from demandops.data.adapters.base import DatasetAdapter

logger = structlog.get_logger()

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def _atomic_download(url: str, dest: Path) -> None:
    """Download to a temp file, then atomically rename on success."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urlretrieve(url, tmp)
        tmp.rename(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class TaxiAdapter(DatasetAdapter):
    """NYC Yellow Taxi: monthly parquets from the TLC CDN."""

    name = "taxi"

    def download(self, raw_dir: Path, months: list[str]) -> list[Path]:
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for month in months:
            filename = f"yellow_tripdata_{month}.parquet"
            dest = raw_dir / filename
            if dest.exists():
                logger.info("file_exists_skipping", path=str(dest))
            else:
                url = f"{TLC_BASE_URL}/{filename}"
                logger.info("downloading", url=url, dest=str(dest))
                _atomic_download(url, dest)
                logger.info("download_complete", path=str(dest))
            paths.append(dest)
        return paths

    def download_zones(self, zones_path: Path) -> Path:
        """Download TLC zone lookup CSV. Idempotent."""
        if zones_path.exists():
            logger.info("file_exists_skipping", path=str(zones_path))
            return zones_path
        logger.info("downloading_zones", url=ZONES_URL, dest=str(zones_path))
        zones_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_download(ZONES_URL, zones_path)
        return zones_path

    def prepare_hourly_history(
        self,
        raw_dir: Path,
        processed_dir: Path,
        config: dict,
    ) -> tuple[pl.DataFrame, list[int]]:
        months = config["data"]["months"]
        zones_path = Path(config["data"]["zones_path"])

        con = duckdb.connect()

        # Load raw parquet files
        parquet_files = [str(raw_dir / f"yellow_tripdata_{m}.parquet") for m in months]
        logger.info("loading_raw_data", files=parquet_files)
        con.execute(
            "CREATE OR REPLACE TABLE raw_trips AS SELECT * FROM read_parquet(?)",
            [parquet_files],
        )

        raw_count = con.execute("SELECT COUNT(*) FROM raw_trips").fetchone()[0]
        logger.info("raw_rows_loaded", count=raw_count)

        # Filter
        con.execute("""
            CREATE OR REPLACE TABLE filtered_trips AS
            SELECT *
            FROM raw_trips
            WHERE tpep_pickup_datetime IS NOT NULL
              AND tpep_dropoff_datetime IS NOT NULL
              AND PULocationID IS NOT NULL
              AND PULocationID BETWEEN 1 AND 263
              AND total_amount IS NOT NULL
              AND total_amount >= 0
              AND trip_distance IS NOT NULL
              AND trip_distance >= 0
              AND tpep_pickup_datetime >= TIMESTAMP '2023-12-01 00:00:00'
              AND tpep_pickup_datetime < TIMESTAMP '2024-03-01 00:00:00'
        """)

        filtered_count = con.execute("SELECT COUNT(*) FROM filtered_trips").fetchone()[0]
        logger.info("filtered_rows", count=filtered_count, dropped=raw_count - filtered_count)

        # Zone universe
        zone_ids_result = con.execute("""
            SELECT DISTINCT PULocationID AS zone_id
            FROM filtered_trips
            ORDER BY zone_id
        """).fetchall()
        zone_ids = [row[0] for row in zone_ids_result]

        # Aggregate to hourly
        con.execute("""
            CREATE OR REPLACE TABLE hourly_agg AS
            SELECT
                PULocationID AS zone_id,
                DATE_TRUNC('hour', tpep_pickup_datetime) AS hour_ts,
                COUNT(*)::INTEGER AS trip_count,
                AVG(total_amount) AS avg_fare,
                AVG(trip_distance) AS avg_distance
            FROM filtered_trips
            GROUP BY PULocationID, DATE_TRUNC('hour', tpep_pickup_datetime)
        """)

        # Densify grid
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
                COALESCE(a.trip_count, 0)::INTEGER AS trip_count,
                a.avg_fare,
                a.avg_distance
            FROM grid g
            LEFT JOIN hourly_agg a
                ON g.zone_id = a.zone_id AND g.hour_ts = a.hour_ts
        """)

        grid_count = con.execute("SELECT COUNT(*) FROM dense_grid").fetchone()[0]
        logger.info("dense_grid_rows", count=grid_count)

        # Join zone names
        con.execute(f"""
            CREATE OR REPLACE TABLE zones_lookup AS
            SELECT "LocationID" AS zone_id, "Zone" AS zone_name
            FROM read_csv('{zones_path}', auto_detect=true)
        """)

        con.execute("""
            CREATE OR REPLACE TABLE hourly_history AS
            SELECT
                g.zone_id,
                COALESCE(z.zone_name, 'Unknown') AS zone_name,
                g.hour_ts,
                g.trip_count,
                g.avg_fare,
                g.avg_distance
            FROM dense_grid g
            LEFT JOIN zones_lookup z ON g.zone_id = z.zone_id
            ORDER BY g.zone_id, g.hour_ts
        """)

        history_df = con.execute("SELECT * FROM hourly_history").pl()
        con.close()

        logger.info("taxi_history_prepared", rows=len(history_df), zones=len(zone_ids))
        return history_df, zone_ids
