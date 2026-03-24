"""Download NYC TLC Yellow Taxi trip data and zone lookup.

Uses urllib.request.urlretrieve — no progress bar or retry.
Acceptable for V1; the TLC CDN is generally reliable.
For flaky connections, run `make download` again (idempotent).
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import structlog

logger = structlog.get_logger()

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def download_month(month: str, raw_dir: Path) -> Path:
    """Download a single month of yellow taxi data. Idempotent."""
    filename = f"yellow_tripdata_{month}.parquet"
    dest = raw_dir / filename
    if dest.exists():
        logger.info("file_exists_skipping", path=str(dest))
        return dest

    url = f"{TLC_BASE_URL}/{filename}"
    logger.info("downloading", url=url, dest=str(dest))
    raw_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    logger.info("download_complete", path=str(dest))
    return dest


def download_zones(zones_path: Path) -> Path:
    """Download TLC zone lookup CSV. Idempotent."""
    if zones_path.exists():
        logger.info("file_exists_skipping", path=str(zones_path))
        return zones_path

    logger.info("downloading_zones", url=ZONES_URL, dest=str(zones_path))
    zones_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(ZONES_URL, zones_path)
    logger.info("download_complete", path=str(zones_path))
    return zones_path


def download_all(months: list[str], raw_dir: Path, zones_path: Path) -> dict:
    """Download all months + zone lookup. Returns dict with paths."""
    month_paths = [download_month(m, raw_dir) for m in months]
    zone_path = download_zones(zones_path)
    return {"months": month_paths, "zones": zone_path}
