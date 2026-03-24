# demandops-lite V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end demand-prediction pipeline on NYC taxi data — from data contracts through honest baselines to lag-aware one-step-ahead monitored inference — demonstrating ML engineering on CPU.

**Architecture:** DuckDB SQL for aggregation + grid densification, Polars for feature engineering, LightGBM for prediction (clipped to zero), FastAPI for serving with FeatureService reconstructing lag features at request time from a dense zone×hour history grid. Pandera validates at every pipeline boundary. prometheus-client instruments /metrics.

**Tech Stack:** Python 3.11+, DuckDB, Polars, Pandera, LightGBM, MLflow, FastAPI, Pydantic v2, structlog, prometheus-client, pytest, Docker, GitHub Actions

**Spec reference:** The full V1 spec (v7) lives in the conversation that produced this plan. All design decisions are locked.

**Implementation risk flags:**
1. Pandera + Polars ergonomics — smoke-test in Task 5
2. Polars `rolling_mean` + `.over()` composability — smoke-test in Task 6
3. MLflow dependency weight — keep unless install friction is unacceptable

---

## Fixes Applied from Plan Review

| # | Issue | Fix |
|---|-------|-----|
| 1 | Polars `dt.weekday()` returns 1–7 (Mon=1, Sun=7), not 0–6 | Subtract 1 in all Polars paths: `.dt.weekday() - 1`. Schema stays `ge=0, le=6`. |
| 2 | `shift(1).rolling_mean(24).over("zone_id")` won't compose correctly | Primary: `group_by("zone_id").map_groups()` with explicit per-group rolling. Fallback approach removed — `map_groups` is unambiguous. |
| 3 | `is_weekend` inconsistent: Polars `>=5` on 1–7 includes Friday | After weekday fix (now 0–6), `>=5` correctly means Sat(5)+Sun(6) in both Polars and Python |
| 4 | LightGBM model loading via `_Booster` internal attribute | Save/load via `joblib`. Add `joblib` to deps. |
| 5 | `conftest.py` fixture duplicates `prepare.py` logic | Extract shared `engineer_features()` and import in both conftest and prepare |
| 6 | `FEATURE_COLUMNS` defined in train.py, consumed everywhere | Move to `demandops/features.py` shared constants module |
| 7 | Clipping stats approximate (checking `==0.0`) | Add `predict_raw()` to LightGBMModel; compute clip count externally |
| 8 | Module-level globals for DI in routes.py | Use `app.state` for dependency injection |
| 9 | `test_client` fixture not in conftest from the start | Serving fixtures in `tests/serving_conftest.py`, loaded only on Day 3 |
| 10 | No test for `prepare.py` itself | Add integration test with tiny synthetic raw parquet |
| 11 | `urlretrieve` has no progress/retry | Acceptable for V1; note in code comment |
| 12 | Pandera DataFrameModel may not support Polars type annotations | Fallback to `DataFrameSchema` functional API ready |
| 13 | sMAPE formula correct | No change needed |
| 14 | Docker `pip install -e .` wrong for containers | Use non-editable `pip install .` with proper two-stage build |
| 15 | `FeatureService._lookup` datetime key precision vs Pydantic timezone-aware | Normalize to naive UTC in `get_features()` before lookup |
| 16 | Prometheus registry pollution across tests | Add `autouse` cleanup fixture in serving_conftest.py |
| 17 | MLflow may drop transitive pandas dependency | Comment in pyproject.toml; low risk for V1 |

---

## Day 1: Data Pipeline (Tasks 1–9)

**Done when:** `make download && make prepare` produces `zone_universe.json` + both validated parquets. Grid is dense. Splits have no gaps. All test_data.py tests pass.

---

### Task 1: Repository Skeleton

**Files:**
- Create: `.gitignore`
- Create: `.dockerignore`
- Create: `demandops/__init__.py`
- Create: `demandops/data/__init__.py`
- Create: `demandops/models/__init__.py`
- Create: `demandops/training/__init__.py`
- Create: `demandops/serving/__init__.py`
- Create: `demandops/monitoring/__init__.py`
- Create: `tests/__init__.py`
- Create: directories for `data/raw/`, `data/processed/`, `artifacts/models/`, `artifacts/reports/`, `scripts/`, `notebooks/`, `docs/`, `docker/`, `configs/`, `.github/workflows/`

**Step 1: Initialize git repo**

```bash
cd /Users/zenith/Desktop/demandops-lite
git init
```

**Step 2: Create .gitignore**

```gitignore
# Data
data/raw/*.parquet
data/processed/*.parquet

# Artifacts
artifacts/models/*.txt
artifacts/models/*.bin
artifacts/models/*.joblib
artifacts/models/feature_schema.json
artifacts/zone_universe.json
artifacts/reports/*.json

# MLflow
mlruns/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
```

**Step 3: Create .dockerignore**

```dockerignore
.git
.gitignore
__pycache__
*.pyc
.venv
venv
mlruns
data/raw
notebooks
tests
docs
.github
*.md
!README.md
```

**Step 4: Create directory structure and __init__.py files**

```bash
mkdir -p data/raw data/processed artifacts/models artifacts/reports scripts notebooks docs docker configs .github/workflows
touch demandops/__init__.py
touch demandops/data/__init__.py
touch demandops/models/__init__.py
touch demandops/training/__init__.py
touch demandops/serving/__init__.py
touch demandops/monitoring/__init__.py
touch tests/__init__.py
```

All `__init__.py` files are empty.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: initialize repo skeleton with directory structure"
```

---

### Task 2: pyproject.toml + Install

**Files:**
- Create: `pyproject.toml`

**Step 1: Write pyproject.toml**

```toml
[project]
name = "demandops-lite"
version = "0.1.0"
description = "End-to-end demand prediction pipeline with ML lifecycle best practices"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.1.0",
    "polars>=1.0.0",
    "pandera[polars]>=0.21.0",
    "lightgbm>=4.5.0",
    "mlflow>=2.16.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "pyyaml>=6.0",
    "structlog>=24.0.0",
    "prometheus-client>=0.21.0",
    "numpy>=1.26.0",
    "httpx>=0.27.0",
    "pyarrow>=17.0.0",
    "joblib>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.9.0",
]

[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 100
```

Notes:
- `pandera[polars]` includes Polars support extra
- `joblib` added for model serialization (fix #4)
- `pandas` and `scikit-learn` NOT listed — add only if a direct import requires them
- MLflow pulls pandas transitively

**Step 2: Create virtual environment and install**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 3: Verify key imports work**

```bash
python -c "import duckdb, polars, pandera, lightgbm, mlflow, fastapi, structlog, prometheus_client, joblib; print('All imports OK')"
```

**Step 4: Smoke-test Pandera + Polars**

```bash
python -c "
import polars as pl
import pandera.polars as pa

schema = pa.DataFrameSchema({
    'x': pa.Column(pl.Int64, checks=pa.Check.ge(0)),
})
df = pl.DataFrame({'x': [1, 2, 3]})
schema.validate(df)
print('Pandera + Polars OK')
"
```

If `DataFrameSchema` works but `DataFrameModel` (class-based API) doesn't, use the functional `DataFrameSchema` approach throughout. Document the decision.

**Step 5: Smoke-test Polars weekday**

```bash
python -c "
import polars as pl
from datetime import datetime
df = pl.DataFrame({'ts': [datetime(2024, 1, 1)]})  # Monday
result = df.select(pl.col('ts').dt.weekday()).item()
print(f'Polars weekday for Monday 2024-01-01: {result}')
# Expected: 1 (Polars uses 1=Mon, 7=Sun)
# Our convention: 0=Mon, 6=Sun → subtract 1
assert result == 1, f'Expected 1, got {result}'
print('Weekday convention confirmed: subtract 1 from Polars weekday')
"
```

**Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml with all V1 dependencies"
```

---

### Task 3: Configuration + Shared Constants

**Files:**
- Create: `configs/default.yaml`
- Create: `demandops/features.py`

**Step 1: Write demandops/features.py (fix #6: shared constants)**

```python
"""Shared feature definitions. Single source of truth for column order.

Every module that needs feature column names or indices imports from here.
If this list changes, all consumers update automatically.
"""

from __future__ import annotations

# Canonical feature column order. Used by:
# - training/train.py (extract from DataFrame)
# - training/evaluate.py (extract from DataFrame)
# - serving/feature_service.py (build feature dict)
# - serving/routes.py (build numpy array)
# - models/baselines.py (column index constants)
# - feature_schema.json (persisted artifact)
FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "zone_id",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
]

TARGET_COLUMN = "trip_count"

# Column indices for models that need positional access (baselines)
IDX_HOUR_OF_DAY = FEATURE_COLUMNS.index("hour_of_day")
IDX_DAY_OF_WEEK = FEATURE_COLUMNS.index("day_of_week")
IDX_IS_WEEKEND = FEATURE_COLUMNS.index("is_weekend")
IDX_MONTH = FEATURE_COLUMNS.index("month")
IDX_ZONE_ID = FEATURE_COLUMNS.index("zone_id")
IDX_LAG_1H = FEATURE_COLUMNS.index("lag_1h")
IDX_LAG_24H = FEATURE_COLUMNS.index("lag_24h")
IDX_LAG_168H = FEATURE_COLUMNS.index("lag_168h")
IDX_ROLLING_MEAN_24H = FEATURE_COLUMNS.index("rolling_mean_24h")
```

**Step 2: Write configs/default.yaml**

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  zones_path: data/zones.csv
  months: ["2023-12", "2024-01", "2024-02"]
  warmup_months: ["2023-12"]
  target_column: trip_count

features:
  lag_hours: [1, 24, 168]
  rolling_windows: [24]
  categorical: [zone_id]
  numerical: [hour_of_day, day_of_week, is_weekend, month,
              lag_1h, lag_24h, lag_168h, rolling_mean_24h]

split:
  train_start: "2024-01-01T00:00:00"
  train_end: "2024-02-01T00:00:00"
  val_end: "2024-02-15T00:00:00"
  test_end: "2024-03-01T00:00:00"

models:
  slot_mean:
    name: slot_mean
  seasonal_naive:
    name: seasonal_naive
  lightgbm:
    name: lightgbm
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6
    num_leaves: 31
    min_child_samples: 20
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 50
    random_state: 42
    num_threads: -1

mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "demandops-v1"

artifacts:
  models_dir: artifacts/models
  reports_dir: artifacts/reports
  feature_schema_path: artifacts/models/feature_schema.json
  zone_universe_path: artifacts/zone_universe.json

serving:
  host: "0.0.0.0"
  port: 8001
  model_name: lightgbm
  history_path: data/processed/hourly_history.parquet
  feature_path: data/processed/features.parquet
  feature_schema_path: artifacts/models/feature_schema.json
  zone_universe_path: artifacts/zone_universe.json
  request_timeout_seconds: 10
```

**Step 3: Commit**

```bash
git add configs/default.yaml demandops/features.py
git commit -m "chore: add config and shared feature constants"
```

---

### Task 4: Data Download Module

**Files:**
- Create: `demandops/data/download.py`
- Create: `scripts/download_data.py`

**Step 1: Write demandops/data/download.py**

```python
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
```

**Step 2: Write scripts/download_data.py**

```python
"""Script entrypoint for data download."""

from pathlib import Path

import yaml

from demandops.data.download import download_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    download_all(
        months=config["data"]["months"],
        raw_dir=Path(config["data"]["raw_dir"]),
        zones_path=Path(config["data"]["zones_path"]),
    )


if __name__ == "__main__":
    main()
```

**Step 3: Verify imports**

```bash
python -c "from demandops.data.download import download_all; print('OK')"
```

**Step 4: Commit**

```bash
git add demandops/data/download.py scripts/download_data.py
git commit -m "feat: add data download module (TLC yellow taxi + zones)"
```

---

### Task 5: Pandera Data Contracts

**Files:**
- Create: `demandops/data/schemas.py`

This is the Pandera + Polars smoke-test. Try `DataFrameModel` first; fall back to `DataFrameSchema` if needed.

**Step 1: Write demandops/data/schemas.py**

Try the class-based API first:

```python
"""Pandera data contracts for pipeline validation.

Weekday convention: 0=Mon, 6=Sun (Python datetime.weekday() convention).
Polars dt.weekday() returns 1–7; we subtract 1 before validation.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class HourlyHistorySchema(pa.DataFrameModel):
    """Schema for hourly_history.parquet — dense zone×hour grid.

    Covers Dec 2023 – Feb 2024. Every (zone_id, hour_ts) pair in the
    zone universe has exactly one row. trip_count is 0 for no-demand hours.
    avg_fare and avg_distance are nullable (null for zero-demand hours).
    """

    zone_id: int = pa.Field(ge=1, le=263)
    zone_name: str = pa.Field(nullable=False)
    hour_ts: pl.Datetime = pa.Field(nullable=False)
    trip_count: int = pa.Field(ge=0)
    avg_fare: float = pa.Field(nullable=True)
    avg_distance: float = pa.Field(nullable=True)


class FeatureSchema(pa.DataFrameModel):
    """Schema for features.parquet — model-ready features, Jan–Feb only.

    No December rows. All lags populated (dense grid guarantees this).
    No nulls in any column. day_of_week uses 0=Mon, 6=Sun convention.
    """

    zone_id: int = pa.Field(ge=1, le=263)
    hour_ts: pl.Datetime = pa.Field(nullable=False)
    trip_count: int = pa.Field(ge=0)
    hour_of_day: int = pa.Field(ge=0, le=23)
    day_of_week: int = pa.Field(ge=0, le=6)
    is_weekend: int = pa.Field(ge=0, le=1)
    month: int = pa.Field(ge=1, le=12)
    lag_1h: float = pa.Field(nullable=False)
    lag_24h: float = pa.Field(nullable=False)
    lag_168h: float = pa.Field(nullable=False)
    rolling_mean_24h: float = pa.Field(nullable=False)


class PredictionOutputSchema(pa.DataFrameModel):
    """Schema for prediction output validation."""

    zone_id: int = pa.Field(ge=1, le=263)
    hour_ts: pl.Datetime = pa.Field(nullable=False)
    predicted_count: float = pa.Field(ge=0.0)
```

**Step 2: Smoke-test with a minimal DataFrame**

```bash
python -c "
import polars as pl
from demandops.data.schemas import HourlyHistorySchema
from datetime import datetime

df = pl.DataFrame({
    'zone_id': [1],
    'zone_name': ['Test'],
    'hour_ts': [datetime(2024, 1, 1)],
    'trip_count': [5],
    'avg_fare': [12.5],
    'avg_distance': [2.1],
})
HourlyHistorySchema.validate(df)
print('HourlyHistorySchema OK')
"
```

**If DataFrameModel fails**, replace with functional API:

```python
HourlyHistorySchema = pa.DataFrameSchema({
    "zone_id": pa.Column(pl.Int64, checks=[pa.Check.ge(1), pa.Check.le(263)]),
    "zone_name": pa.Column(pl.Utf8, nullable=False),
    "hour_ts": pa.Column(pl.Datetime, nullable=False),
    "trip_count": pa.Column(pl.Int64, checks=pa.Check.ge(0)),
    "avg_fare": pa.Column(pl.Float64, nullable=True),
    "avg_distance": pa.Column(pl.Float64, nullable=True),
})
```

Document which approach worked in a code comment.

**Step 3: Commit**

```bash
git add demandops/data/schemas.py
git commit -m "feat: add Pandera data contracts for pipeline validation"
```

---

### Task 6: Data Preparation

**Files:**
- Create: `demandops/data/prepare.py`
- Create: `scripts/prepare_data.py`

Most complex module. Produces `zone_universe.json`, `hourly_history.parquet`, `features.parquet`.

**Critical fixes applied:**
- Fix #1: `dt.weekday() - 1` for 0-based day_of_week
- Fix #2: Proper rolling mean computation per group
- Fix #3: `is_weekend` uses corrected weekday (>=5 means Sat/Sun)
- Fix #5: Feature engineering extracted to reusable `engineer_features()`

**Step 1: Write demandops/data/prepare.py**

```python
"""Data preparation pipeline: raw parquet → dense grid → features.

Pipeline steps:
1. Load 3 months of raw parquet via DuckDB
2. Filter nulls, negative fares, out-of-range dates
3. Determine zone universe, save as JSON
4. Aggregate to hourly trip counts per zone (DuckDB)
5. Densify grid: cartesian product of zones × hours (DuckDB)
6. Join zone lookup for zone names
7. Save hourly_history.parquet (Dec–Feb, dense)
8. Feature engineering with Polars (lags, temporal, rolling)
9. Drop December rows
10. Save features.parquet (Jan–Feb, model-ready)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import structlog

logger = structlog.get_logger()


def engineer_features(
    df: pl.DataFrame,
    lag_hours: list[int],
    rolling_windows: list[int],
) -> pl.DataFrame:
    """Add temporal features, lags, and rolling means.

    Operates on the dense grid so all lags are guaranteed non-null
    after the warm-up period.

    IMPORTANT: Polars dt.weekday() returns 1=Mon through 7=Sun.
    We subtract 1 to get 0=Mon through 6=Sun, matching Python's
    datetime.weekday() convention. This ensures train-serve parity
    with FeatureService which uses Python datetime.
    """
    # Must be sorted per zone for correct lag/rolling computation
    df = df.sort(["zone_id", "hour_ts"])

    # Temporal features (fix #1: subtract 1 from Polars weekday)
    polars_weekday = pl.col("hour_ts").dt.weekday() - 1  # 0=Mon, 6=Sun
    df = df.with_columns(
        pl.col("hour_ts").dt.hour().alias("hour_of_day"),
        polars_weekday.alias("day_of_week"),
        (polars_weekday >= 5).cast(pl.Int64).alias("is_weekend"),  # fix #3: Sat=5, Sun=6
        pl.col("hour_ts").dt.month().alias("month"),
    )

    # Lag features (per zone, using shift on sorted data)
    for lag in lag_hours:
        df = df.with_columns(
            pl.col("trip_count")
            .shift(lag)
            .over("zone_id")
            .cast(pl.Float64)
            .alias(f"lag_{lag}h")
        )

    # Rolling mean (fix #2: group_by.map_groups for unambiguous per-group rolling)
    # We need: for each row, the mean of trip_count at hours [t-24, t-1]
    # Using map_groups because rolling_mean().over() has version-dependent
    # behavior — rolling is a window function that operates on physical row
    # positions, and .over() partitioning doesn't guarantee correct boundaries.
    for window in rolling_windows:
        def _add_rolling(group_df: pl.DataFrame, w: int = window) -> pl.DataFrame:
            return group_df.with_columns(
                pl.col("trip_count")
                .shift(1)
                .rolling_mean(window_size=w, min_periods=w)
                .alias(f"rolling_mean_{w}h")
            )

        df = df.group_by("zone_id", maintain_order=True).map_groups(_add_rolling)

    return df


def prepare(
    raw_dir: Path,
    processed_dir: Path,
    zones_path: Path,
    zone_universe_path: Path,
    months: list[str],
    lag_hours: list[int],
    rolling_windows: list[int],
) -> dict[str, Path]:
    """Run full preparation pipeline.

    Returns:
        Dict with output paths: history_path, features_path, zone_universe_path
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    zone_universe_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Step 1: Load raw parquet files
    parquet_files = [str(raw_dir / f"yellow_tripdata_{m}.parquet") for m in months]
    logger.info("loading_raw_data", files=parquet_files)

    con.execute("""
        CREATE OR REPLACE TABLE raw_trips AS
        SELECT * FROM read_parquet(?)
    """, [parquet_files])

    raw_count = con.execute("SELECT COUNT(*) FROM raw_trips").fetchone()[0]
    logger.info("raw_rows_loaded", count=raw_count)

    # Step 2: Filter
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

    # Step 3: Zone universe
    zone_ids_result = con.execute("""
        SELECT DISTINCT PULocationID AS zone_id
        FROM filtered_trips
        ORDER BY zone_id
    """).fetchall()
    zone_ids = [row[0] for row in zone_ids_result]

    zone_universe = {
        "zone_ids": zone_ids,
        "n_zones": len(zone_ids),
        "source": "raw_trips distinct PULocationIDs",
    }
    zone_universe_path.write_text(json.dumps(zone_universe, indent=2))
    logger.info("zone_universe_saved", n_zones=len(zone_ids), path=str(zone_universe_path))

    # Step 4: Aggregate to hourly
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

    # Step 5: Densify grid
    # Register zone_ids as a DuckDB value list
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

    # Step 6: Join zone names
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

    # Step 7: Save hourly_history.parquet
    history_path = processed_dir / "hourly_history.parquet"
    history_df = con.execute("SELECT * FROM hourly_history").pl()
    history_df.write_parquet(history_path)
    logger.info("hourly_history_saved", path=str(history_path), rows=len(history_df))

    con.close()

    # Step 8: Feature engineering (Polars)
    features_df = engineer_features(history_df, lag_hours, rolling_windows)

    # Step 9: Drop December rows
    features_df = features_df.filter(pl.col("hour_ts") >= datetime(2024, 1, 1))

    # Drop rows with null lags or rolling means (shouldn't happen for Jan+
    # since Dec provides warm-up, but be explicit)
    lag_cols = [f"lag_{h}h" for h in lag_hours]
    rolling_cols = [f"rolling_mean_{w}h" for w in rolling_windows]
    features_df = features_df.drop_nulls(subset=lag_cols + rolling_cols)

    logger.info("features_after_cleanup", rows=len(features_df))

    # Step 10: Save features.parquet
    features_path = processed_dir / "features.parquet"
    features_df.write_parquet(features_path)
    logger.info("features_saved", path=str(features_path), rows=len(features_df))

    return {
        "history_path": history_path,
        "features_path": features_path,
        "zone_universe_path": zone_universe_path,
    }
```

**Step 2: Write scripts/prepare_data.py**

```python
"""Script entrypoint for data preparation."""

from pathlib import Path

import yaml

from demandops.data.prepare import prepare


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    prepare(
        raw_dir=Path(config["data"]["raw_dir"]),
        processed_dir=Path(config["data"]["processed_dir"]),
        zones_path=Path(config["data"]["zones_path"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
        months=config["data"]["months"],
        lag_hours=config["features"]["lag_hours"],
        rolling_windows=config["features"]["rolling_windows"],
    )


if __name__ == "__main__":
    main()
```

**Step 3: Smoke-test the rolling mean logic on a tiny DataFrame**

```bash
python -c "
import polars as pl
from datetime import datetime, timedelta

# 1 zone, 30 hours — check rolling mean
rows = []
for i in range(30):
    rows.append({'zone_id': 1, 'zone_name': 'Test',
                 'hour_ts': datetime(2024, 1, 1) + timedelta(hours=i),
                 'trip_count': i, 'avg_fare': None, 'avg_distance': None})
df = pl.DataFrame(rows)

from demandops.data.prepare import engineer_features
result = engineer_features(df, lag_hours=[1], rolling_windows=[3])
# At hour 4 (index 4), rolling_mean_3h should be mean of hours 1,2,3 = mean(1,2,3) = 2.0
row4 = result.filter(pl.col('hour_ts') == datetime(2024, 1, 1, 4))
rm = row4['rolling_mean_3h'].item()
print(f'Rolling mean at hour 4: {rm} (expected ~2.0)')
assert abs(rm - 2.0) < 0.1, f'Rolling mean wrong: {rm}'
print('Rolling mean OK')
"
```

This is the critical smoke test for fix #2. If this fails, the `rolling_mean` + `over()` approach needs rework (try `group_by("zone_id").map_groups()` instead).

**Step 4: Commit**

```bash
git add demandops/data/prepare.py scripts/prepare_data.py
git commit -m "feat: add data preparation (zone universe + dense grid + features)"
```

---

### Task 7: Temporal Split

**Files:**
- Create: `demandops/data/splits.py`

**Step 1: Write demandops/data/splits.py**

```python
"""Temporal split using half-open intervals.

Every boundary uses >= start and < end. No ambiguity.
train_end == val_start, val_end == test_start: no gaps, no overlaps.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl


def temporal_split(
    df: pl.DataFrame,
    train_start: str | datetime,
    train_end: str | datetime,
    val_end: str | datetime,
    test_end: str | datetime,
    ts_column: str = "hour_ts",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame into train/val/test using half-open intervals.

    Args:
        df: DataFrame with a timestamp column
        train_start: Inclusive start of training window
        train_end: Exclusive end of training / inclusive start of validation
        val_end: Exclusive end of validation / inclusive start of test
        test_end: Exclusive end of test window
        ts_column: Name of the timestamp column

    Returns:
        (train, val, test) DataFrames
    """
    if isinstance(train_start, str):
        train_start = datetime.fromisoformat(train_start)
    if isinstance(train_end, str):
        train_end = datetime.fromisoformat(train_end)
    if isinstance(val_end, str):
        val_end = datetime.fromisoformat(val_end)
    if isinstance(test_end, str):
        test_end = datetime.fromisoformat(test_end)

    ts = pl.col(ts_column)

    train = df.filter((ts >= train_start) & (ts < train_end))
    val = df.filter((ts >= train_end) & (ts < val_end))
    test = df.filter((ts >= val_end) & (ts < test_end))

    return train, val, test


def split_from_config(
    df: pl.DataFrame, config: dict, ts_column: str = "hour_ts"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split using config dict with split.train_start etc."""
    split_cfg = config["split"]
    return temporal_split(
        df,
        train_start=split_cfg["train_start"],
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
        test_end=split_cfg["test_end"],
        ts_column=ts_column,
    )
```

**Step 2: Commit**

```bash
git add demandops/data/splits.py
git commit -m "feat: add half-open temporal split"
```

---

### Task 8: Test Fixtures + test_data.py

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`

**Critical:** conftest.py uses `engineer_features()` from `prepare.py` (fix #5: no duplicated logic). Also includes serving test fixtures from the start (fix #9).

**Step 1: Write tests/conftest.py**

```python
"""Shared test fixtures. Deterministic, dense, synthetic data.

Uses engineer_features() from prepare.py — no duplicated logic.

Serving fixtures use lazy imports (fix #9) so that Day 1 tests
can run before the serving modules exist. The imports execute only
when a test actually requests the fixture.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from demandops.data.prepare import engineer_features

SEED = 42


# ──────────────────────────────────────────────
# Data pipeline fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def zone_universe() -> dict:
    """Small zone universe for testing: 3 zones."""
    return {"zone_ids": [1, 2, 3], "n_zones": 3, "source": "test fixture"}


@pytest.fixture
def zone_universe_path(tmp_path: Path, zone_universe: dict) -> Path:
    path = tmp_path / "zone_universe.json"
    path.write_text(json.dumps(zone_universe))
    return path


@pytest.fixture
def dense_history_df(zone_universe: dict) -> pl.DataFrame:
    """Dense zone×hour grid: 3 zones × Dec 2023 – Feb 2024.

    Deterministic trip_count: zone_id * hour_of_day + day_of_week.
    Zone 3 has zero demand late night (hours 0-4).
    """
    rng = np.random.RandomState(SEED)
    zone_ids = zone_universe["zone_ids"]

    start = datetime(2023, 12, 1, 0, 0, 0)
    end = datetime(2024, 2, 29, 23, 0, 0)  # 2024 is leap year

    hours = []
    ts = start
    while ts <= end:
        hours.append(ts)
        ts += timedelta(hours=1)

    rows = []
    for zone_id in zone_ids:
        for hour_ts in hours:
            # Deterministic demand pattern using Python weekday (0=Mon)
            base = zone_id * hour_ts.hour + hour_ts.weekday()
            if zone_id == 3 and hour_ts.hour < 5:
                trip_count = 0
                avg_fare = None
                avg_distance = None
            else:
                trip_count = max(0, base + rng.randint(-2, 5))
                avg_fare = float(round(10.0 + trip_count * 0.5 + rng.random() * 5, 2))
                avg_distance = float(round(1.0 + rng.random() * 3, 2))

            rows.append({
                "zone_id": zone_id,
                "zone_name": f"Zone {zone_id}",
                "hour_ts": hour_ts,
                "trip_count": trip_count,
                "avg_fare": avg_fare,
                "avg_distance": avg_distance,
            })

    return pl.DataFrame(rows).cast({
        "zone_id": pl.Int64,
        "trip_count": pl.Int64,
        "hour_ts": pl.Datetime("us"),
    })


@pytest.fixture
def features_df(dense_history_df: pl.DataFrame) -> pl.DataFrame:
    """Features DataFrame: Jan–Feb only, with lags and temporal features.

    Uses engineer_features() from prepare.py — no duplicated logic (fix #5).
    """
    df = engineer_features(dense_history_df, lag_hours=[1, 24, 168], rolling_windows=[24])

    # Drop December rows
    df = df.filter(pl.col("hour_ts") >= datetime(2024, 1, 1))

    # Drop rows with null lags (warm-up period)
    df = df.drop_nulls(
        subset=["lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h"]
    )

    return df


@pytest.fixture
def split_config() -> dict:
    """Split configuration matching default.yaml."""
    return {
        "split": {
            "train_start": "2024-01-01T00:00:00",
            "train_end": "2024-02-01T00:00:00",
            "val_end": "2024-02-15T00:00:00",
            "test_end": "2024-03-01T00:00:00",
        }
    }


@pytest.fixture
def history_parquet_path(tmp_path: Path, dense_history_df: pl.DataFrame) -> Path:
    path = tmp_path / "hourly_history.parquet"
    dense_history_df.write_parquet(path)
    return path


# ──────────────────────────────────────────────
# Serving fixtures (fix #9: lazy imports — no top-level serving deps)
# All serving module imports are inside fixture bodies so that
# Day 1 pytest collection succeeds before serving code exists.
# ──────────────────────────────────────────────

@pytest.fixture
def feature_schema_path(tmp_path: Path) -> Path:
    from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN
    schema = {
        "columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "categorical": ["zone_id"],
        "numerical": [c for c in FEATURE_COLUMNS if c != "zone_id"],
    }
    path = tmp_path / "feature_schema.json"
    path.write_text(json.dumps(schema))
    return path


@pytest.fixture
def feature_service(
    history_parquet_path: Path,
    feature_schema_path: Path,
    zone_universe_path: Path,
    split_config: dict,
):
    from demandops.serving.feature_service import FeatureService
    return FeatureService(
        history_path=history_parquet_path,
        schema_path=feature_schema_path,
        zone_universe_path=zone_universe_path,
        config=split_config,
    )


@pytest.fixture
def mock_feature_service(zone_universe: dict) -> MagicMock:
    """Mock FeatureService with predictable responses.

    Uses a local FeatureResult dataclass to avoid importing from
    serving module at fixture definition time.
    """
    from demandops.serving.feature_service import FeatureResult

    svc = MagicMock()
    svc.supported_start = datetime(2024, 1, 1)
    svc.supported_end = datetime(2024, 3, 1)
    svc.n_supported_zones = zone_universe["n_zones"]
    svc.zone_universe = set(zone_universe["zone_ids"])
    svc.history = MagicMock()
    svc.history.__len__ = MagicMock(return_value=6624)
    svc.get_zone_name.return_value = "Test Zone"

    def get_features(zone_id, hour_ts):
        if zone_id not in svc.zone_universe:
            return FeatureResult(
                features=None, supported=False,
                warnings=[f"zone_id {zone_id} not in supported zone universe"],
            )
        if hour_ts < svc.supported_start or hour_ts >= svc.supported_end:
            return FeatureResult(
                features=None, supported=False,
                warnings=["hour_ts outside supported range"],
            )
        return FeatureResult(
            features={
                "hour_of_day": hour_ts.hour,
                "day_of_week": hour_ts.weekday(),  # Python: 0=Mon
                "is_weekend": 1 if hour_ts.weekday() >= 5 else 0,
                "month": hour_ts.month,
                "zone_id": zone_id,
                "lag_1h": 5.0,
                "lag_24h": 4.0,
                "lag_168h": 6.0,
                "rolling_mean_24h": 5.0,
            },
            supported=True,
        )

    svc.get_features.side_effect = get_features
    return svc


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.array([42.5])
    return model


@pytest.fixture
def test_app(mock_feature_service, mock_model):
    from fastapi import FastAPI
    from demandops.serving.routes import configure, router
    app = FastAPI()
    app.include_router(router)
    configure(app, mock_feature_service, mock_model, "lightgbm", time.time())
    return app


@pytest.fixture
def test_client(test_app):
    from fastapi.testclient import TestClient
    return TestClient(test_app)
```

**Step 2: Write tests/test_data.py**

```python
"""Tests for data pipeline: schemas, grid completeness, splits, weekday convention."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from demandops.data.schemas import FeatureSchema, HourlyHistorySchema
from demandops.data.splits import temporal_split


class TestHourlyHistorySchema:

    def test_schema_validates(self, dense_history_df: pl.DataFrame) -> None:
        HourlyHistorySchema.validate(dense_history_df)

    def test_includes_december(self, dense_history_df: pl.DataFrame) -> None:
        dec_rows = dense_history_df.filter(pl.col("hour_ts") < datetime(2024, 1, 1))
        assert len(dec_rows) > 0

    def test_trip_count_non_negative(self, dense_history_df: pl.DataFrame) -> None:
        assert dense_history_df["trip_count"].min() >= 0

    def test_zone_name_populated(self, dense_history_df: pl.DataFrame) -> None:
        assert dense_history_df["zone_name"].null_count() == 0

    def test_zero_demand_hours_exist(self, dense_history_df: pl.DataFrame) -> None:
        zero_rows = dense_history_df.filter(pl.col("trip_count") == 0)
        assert len(zero_rows) > 0


class TestGridCompleteness:

    def test_row_count(
        self, dense_history_df: pl.DataFrame, zone_universe: dict
    ) -> None:
        n_zones = zone_universe["n_zones"]
        n_unique_hours = dense_history_df["hour_ts"].n_unique()
        expected = n_zones * n_unique_hours
        assert len(dense_history_df) == expected

    def test_every_zone_has_every_hour(
        self, dense_history_df: pl.DataFrame
    ) -> None:
        counts = dense_history_df.group_by("zone_id").len()
        unique_counts = counts["len"].unique()
        assert len(unique_counts) == 1

    def test_no_duplicate_zone_hour_pairs(
        self, dense_history_df: pl.DataFrame
    ) -> None:
        n_unique = dense_history_df.select(
            pl.struct("zone_id", "hour_ts").n_unique()
        ).item()
        assert n_unique == len(dense_history_df)


class TestFeatureSchema:

    def test_schema_validates(self, features_df: pl.DataFrame) -> None:
        FeatureSchema.validate(features_df)

    def test_no_december_rows(self, features_df: pl.DataFrame) -> None:
        dec = features_df.filter(pl.col("hour_ts") < datetime(2024, 1, 1))
        assert len(dec) == 0

    def test_no_null_lags(self, features_df: pl.DataFrame) -> None:
        for col in ["lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h"]:
            assert features_df[col].null_count() == 0, f"{col} has nulls"

    def test_day_of_week_0_to_6(self, features_df: pl.DataFrame) -> None:
        """day_of_week must be 0–6 (0=Mon, 6=Sun), not 1–7."""
        assert features_df["day_of_week"].min() >= 0
        assert features_df["day_of_week"].max() <= 6

    def test_is_weekend_only_sat_sun(self, features_df: pl.DataFrame) -> None:
        """is_weekend=1 only when day_of_week >= 5 (Sat=5, Sun=6)."""
        weekend_rows = features_df.filter(pl.col("is_weekend") == 1)
        assert weekend_rows["day_of_week"].min() >= 5
        weekday_rows = features_df.filter(pl.col("is_weekend") == 0)
        assert weekday_rows["day_of_week"].max() <= 4

    def test_weekday_matches_python_convention(
        self, features_df: pl.DataFrame
    ) -> None:
        """Spot-check: 2024-01-01 is a Monday → day_of_week should be 0."""
        monday_row = features_df.filter(
            pl.col("hour_ts") == datetime(2024, 1, 1, 12, 0, 0)
        )
        if len(monday_row) > 0:
            assert monday_row["day_of_week"].item() == 0  # Monday = 0


class TestTemporalSplit:

    def test_no_overlap_no_gap(
        self, features_df: pl.DataFrame, split_config: dict
    ) -> None:
        train, val, test = temporal_split(
            features_df, **split_config["split"]
        )
        # Exactly 1 hour gap between last train and first val timestamp
        gap = val["hour_ts"].min() - train["hour_ts"].max()
        assert gap == timedelta(hours=1)
        gap2 = test["hour_ts"].min() - val["hour_ts"].max()
        assert gap2 == timedelta(hours=1)

    def test_half_open_boundaries(
        self, features_df: pl.DataFrame, split_config: dict
    ) -> None:
        train, val, test = temporal_split(
            features_df, **split_config["split"]
        )
        assert train["hour_ts"].max() < datetime(2024, 2, 1)
        assert val["hour_ts"].min() >= datetime(2024, 2, 1)
        assert val["hour_ts"].max() < datetime(2024, 2, 15)
        assert test["hour_ts"].min() >= datetime(2024, 2, 15)
        assert test["hour_ts"].max() < datetime(2024, 3, 1)

    def test_chronological_order(
        self, features_df: pl.DataFrame, split_config: dict
    ) -> None:
        train, val, test = temporal_split(
            features_df, **split_config["split"]
        )
        assert train["hour_ts"].max() < val["hour_ts"].min()
        assert val["hour_ts"].max() < test["hour_ts"].min()

    def test_no_december_in_any_split(
        self, features_df: pl.DataFrame, split_config: dict
    ) -> None:
        train, val, test = temporal_split(
            features_df, **split_config["split"]
        )
        for name, split in [("train", train), ("val", val), ("test", test)]:
            dec = split.filter(pl.col("hour_ts") < datetime(2024, 1, 1))
            assert len(dec) == 0, f"{name} contains December rows"

    def test_test_includes_feb_29(
        self, features_df: pl.DataFrame, split_config: dict
    ) -> None:
        _, _, test = temporal_split(
            features_df, **split_config["split"]
        )
        feb29 = test.filter(
            (pl.col("hour_ts").dt.month() == 2)
            & (pl.col("hour_ts").dt.day() == 29)
        )
        assert len(feb29) > 0

    def test_splits_cover_all_data(
        self, features_df: pl.DataFrame, split_config: dict
    ) -> None:
        train, val, test = temporal_split(
            features_df, **split_config["split"]
        )
        total = len(train) + len(val) + len(test)
        assert total == len(features_df)
```

**Step 3: Run tests**

```bash
pytest tests/test_data.py -v
```

Expected: ALL PASS. If Pandera schema validation fails, switch to functional API and re-run.

**Step 4: Commit**

```bash
git add tests/conftest.py tests/test_data.py
git commit -m "test: add data pipeline tests (schemas, grid, splits, weekday convention)"
```

---

### Task 9: Makefile + Integration Test

**Files:**
- Create: `Makefile`

**Step 1: Write Makefile**

```makefile
.PHONY: download prepare train evaluate benchmark serve test lint clean pipeline

download:
	python scripts/download_data.py

prepare:
	python scripts/prepare_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

benchmark:
	python scripts/benchmark.py

serve:
	uvicorn demandops.serving.app:app --host 0.0.0.0 --port 8001 --reload

test:
	pytest tests/ -v

lint:
	ruff check demandops/ tests/
	ruff format --check demandops/ tests/

format:
	ruff check --fix demandops/ tests/
	ruff format demandops/ tests/

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet
	rm -rf artifacts/models/*.txt artifacts/models/*.bin artifacts/models/*.joblib
	rm -rf artifacts/models/feature_schema.json artifacts/zone_universe.json
	rm -rf artifacts/reports/*.json mlruns/
	rm -rf __pycache__ .pytest_cache

pipeline: download prepare train evaluate benchmark
```

**Step 2: Verify `make test` passes**

```bash
make test
```

**Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add Makefile with pipeline targets"
```

---

## Day 2: Models + Training (Tasks 10–15)

**Done when:** Three models trained on fixtures, metrics computed, feature_schema.json produced, clipping count tracked. All test_models.py tests pass. LightGBM predictions non-negative.

---

### Task 10: Model Interface + Registry

**Files:**
- Create: `demandops/models/registry.py`

**Step 1: Write demandops/models/registry.py**

```python
"""Model interface and factory registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DemandModel(ABC):
    """Base class for all demand prediction models."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions. Must return non-negative values."""

    def get_params(self) -> dict:
        """Return model parameters for logging."""
        return {}


MODEL_REGISTRY: dict[str, type[DemandModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def wrapper(cls: type[DemandModel]):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def create_model(name: str, **kwargs: Any) -> DemandModel:
    """Factory: create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
```

**Step 2: Commit**

```bash
git add demandops/models/registry.py
git commit -m "feat: add model interface and registry"
```

---

### Task 11: Baselines

**Files:**
- Create: `demandops/models/baselines.py`
- Create: `tests/test_models.py`

**Step 1: Write tests/test_models.py (baseline tests)**

```python
"""Tests for model implementations."""

from __future__ import annotations

import numpy as np
import pytest

from demandops.features import (
    IDX_DAY_OF_WEEK,
    IDX_HOUR_OF_DAY,
    IDX_LAG_168H,
)
from demandops.models.registry import create_model


def _make_feature_array(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Helper: generate a random feature array with correct column count."""
    return np.column_stack([
        rng.randint(0, 24, n),             # hour_of_day
        rng.randint(0, 7, n),              # day_of_week (0=Mon, 6=Sun)
        rng.randint(0, 2, n),              # is_weekend
        rng.randint(1, 3, n),              # month
        rng.randint(1, 4, n),              # zone_id
        rng.poisson(3, n).astype(float),   # lag_1h
        rng.poisson(3, n).astype(float),   # lag_24h
        rng.poisson(3, n).astype(float),   # lag_168h
        rng.poisson(3, n).astype(float),   # rolling_mean_24h
    ]).astype(float)


class TestSlotMean:

    def test_predictions_non_negative(self) -> None:
        model = create_model("slot_mean")
        rng = np.random.RandomState(42)
        X = _make_feature_array(rng, 100)
        y = rng.poisson(5, 100).astype(float)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert (preds >= 0).all()

    def test_predictions_are_means_of_slots(self) -> None:
        model = create_model("slot_mean")
        # 2 rows for hour=0/dow=0 with values 4,6 → mean=5
        X = np.array([
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # hour=0, dow=0
            [0, 0, 0, 1, 1, 0, 0, 0, 0],  # hour=0, dow=0
            [12, 3, 0, 1, 1, 0, 0, 0, 0], # hour=12, dow=3
        ], dtype=float)
        y = np.array([4.0, 6.0, 10.0])
        model.fit(X, y)
        pred = model.predict(np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=float))
        assert abs(pred[0] - 5.0) < 1e-6


class TestSeasonalNaive:

    def test_predictions_equal_lag168(self) -> None:
        model = create_model("seasonal_naive")
        X = np.array([
            [0, 0, 0, 1, 1, 5.0, 10.0, 42.0, 8.0],
            [12, 3, 0, 1, 2, 3.0, 7.0, 99.0, 6.0],
        ], dtype=float)
        y = np.array([0.0, 0.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert abs(preds[0] - 42.0) < 1e-6
        assert abs(preds[1] - 99.0) < 1e-6

    def test_predictions_non_negative(self) -> None:
        model = create_model("seasonal_naive")
        X = np.array([[0, 0, 0, 1, 1, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        y = np.array([0.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()
```

**Step 2: Run to verify tests fail**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL — `slot_mean` and `seasonal_naive` not registered.

**Step 3: Write demandops/models/baselines.py**

```python
"""Baseline models: Historical Slot Mean and Seasonal Naive."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from demandops.features import IDX_DAY_OF_WEEK, IDX_HOUR_OF_DAY, IDX_LAG_168H
from demandops.models.registry import DemandModel, register_model


@register_model("slot_mean")
class HistoricalSlotMean(DemandModel):
    """Predict the historical mean for each (hour_of_day, day_of_week) slot.

    Predictions are naturally non-negative (mean of non-negative counts).
    """

    name = "slot_mean"

    def __init__(self, **kwargs: Any) -> None:
        self._slot_means: dict[tuple[int, int], float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        slot_sums: dict[tuple[int, int], float] = defaultdict(float)
        slot_counts: dict[tuple[int, int], int] = defaultdict(int)

        for i in range(len(X)):
            key = (int(X[i, IDX_HOUR_OF_DAY]), int(X[i, IDX_DAY_OF_WEEK]))
            slot_sums[key] += y[i]
            slot_counts[key] += 1

        self._slot_means = {
            k: slot_sums[k] / slot_counts[k] for k in slot_sums
        }
        self._global_mean = float(np.mean(y)) if len(y) > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.empty(len(X))
        for i in range(len(X)):
            key = (int(X[i, IDX_HOUR_OF_DAY]), int(X[i, IDX_DAY_OF_WEEK]))
            preds[i] = self._slot_means.get(key, self._global_mean)
        return preds

    def get_params(self) -> dict:
        return {"n_slots": len(self._slot_means)}


@register_model("seasonal_naive")
class SeasonalNaive(DemandModel):
    """Predict using lag_168h (same hour, same day, one week ago).

    Predictions are exact historical counts — non-negative by definition.
    """

    name = "seasonal_naive"

    def __init__(self, **kwargs: Any) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, IDX_LAG_168H].copy()

    def get_params(self) -> dict:
        return {"lag_column": "lag_168h"}
```

**Step 4: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: ALL PASS.

**Step 5: Commit**

```bash
git add demandops/models/baselines.py tests/test_models.py
git commit -m "feat: add baseline models (slot mean + seasonal naive) with tests"
```

---

### Task 12: LightGBM Model

**Files:**
- Create: `demandops/models/lightgbm_model.py`

**Step 1: Add LightGBM tests to tests/test_models.py**

Append to existing file:

```python
class TestLightGBM:

    @pytest.fixture
    def trained_lgbm(self) -> tuple:
        rng = np.random.RandomState(42)
        X = _make_feature_array(rng, 200)
        y = rng.poisson(3, 200).astype(float)

        model = create_model(
            "lightgbm", n_estimators=10, num_threads=1,
            random_state=42, verbose=-1,
        )
        model.fit(X[:150], y[:150], eval_set=(X[150:], y[150:]))
        return model, X

    def test_predictions_non_negative(self, trained_lgbm: tuple) -> None:
        model, X = trained_lgbm
        preds = model.predict(X)
        assert (preds >= 0).all(), f"Min prediction: {preds.min()}"

    def test_predictions_near_zero_targets(self) -> None:
        """Even with near-zero targets, predictions are non-negative."""
        rng = np.random.RandomState(42)
        n = 200
        X = _make_feature_array(rng, n)
        # Override lag columns with zeros to push predictions near zero
        X[:, 5:9] = 0.0
        y = rng.uniform(-0.1, 0.5, n).clip(0)

        model = create_model(
            "lightgbm", n_estimators=10, num_threads=1,
            random_state=42, verbose=-1,
        )
        model.fit(X[:150], y[:150], eval_set=(X[150:], y[150:]))
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_predict_raw_returns_unclipped(self, trained_lgbm: tuple) -> None:
        """predict_raw() returns unclipped values (fix #7)."""
        model, X = trained_lgbm
        raw = model.predict_raw(X)
        clipped = model.predict(X)
        # Clipped should be >= 0, raw may have negatives
        assert (clipped >= 0).all()
        # Where raw >= 0, clipped == raw
        mask = raw >= 0
        np.testing.assert_array_almost_equal(clipped[mask], raw[mask])

    def test_deterministic_with_seed(self, trained_lgbm: tuple) -> None:
        model, X = trained_lgbm
        preds1 = model.predict(X[:10])
        preds2 = model.predict(X[:10])
        np.testing.assert_array_equal(preds1, preds2)

    def test_save_and_load_roundtrip(self, trained_lgbm: tuple, tmp_path) -> None:
        """Model survives joblib save/load (fix #4)."""
        import joblib
        model, X = trained_lgbm
        preds_before = model.predict(X[:10])

        path = tmp_path / "model.joblib"
        model.save(path)

        loaded = create_model("lightgbm", num_threads=1, verbose=-1)
        loaded.load(path)
        preds_after = loaded.predict(X[:10])

        np.testing.assert_array_almost_equal(preds_before, preds_after)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py::TestLightGBM -v
```

Expected: FAIL — `lightgbm` not registered.

**Step 3: Write demandops/models/lightgbm_model.py**

```python
"""LightGBM demand prediction model with non-negative clipping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from demandops.models.registry import DemandModel, register_model


@register_model("lightgbm")
class LightGBMModel(DemandModel):
    """LightGBM regressor with predictions clipped to zero.

    predict() returns np.clip(raw_predictions, 0.0, None).
    predict_raw() returns unclipped predictions for tracking clip stats.
    save()/load() use joblib for reliable serialization (fix #4).
    """

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        num_threads: int = -1,
        verbose: int = -1,
        **kwargs: Any,
    ) -> None:
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": num_threads,
            "verbose": verbose,
        }
        self._early_stopping_rounds = early_stopping_rounds
        self._model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = lgb.LGBMRegressor(**self._params)

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["callbacks"] = [
                lgb.early_stopping(self._early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self._model.fit(X, y, **fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with non-negative clipping."""
        return np.clip(self.predict_raw(X), 0.0, None)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict without clipping. Use for tracking clip statistics."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        """Save model via joblib (reliable round-trip, fix #4)."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        """Load model from joblib file."""
        self._model = joblib.load(path)

    def get_params(self) -> dict:
        params = dict(self._params)
        if self._model is not None and hasattr(self._model, "best_iteration_"):
            params["best_iteration"] = self._model.best_iteration_
        return params

    @property
    def feature_importances(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return self._model.feature_importances_

    @property
    def booster(self) -> lgb.LGBMRegressor | None:
        return self._model
```

**Step 4: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: ALL PASS.

**Step 5: Commit**

```bash
git add demandops/models/lightgbm_model.py tests/test_models.py
git commit -m "feat: add LightGBM model with clipping, predict_raw, joblib save/load"
```

---

### Task 13: Training Script

**Files:**
- Create: `demandops/training/train.py`
- Create: `scripts/train.py`

**Step 1: Write demandops/training/train.py**

```python
"""Training pipeline: load features, split, train models, save artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import structlog

from demandops.data.splits import split_from_config
from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN
from demandops.models.registry import create_model

logger = structlog.get_logger()


def train_model(
    features_path: Path,
    config: dict,
    model_name: str,
    models_dir: Path,
    feature_schema_path: Path,
) -> dict[str, Any]:
    """Train a single model and save artifacts."""
    df = pl.read_parquet(features_path)
    train, val, _ = split_from_config(df, config)

    logger.info("split_sizes", train=len(train), val=len(val), model=model_name)

    X_train = train.select(FEATURE_COLUMNS).to_numpy()
    y_train = train[TARGET_COLUMN].to_numpy().astype(float)
    X_val = val.select(FEATURE_COLUMNS).to_numpy()
    y_val = val[TARGET_COLUMN].to_numpy().astype(float)

    model_config = config["models"].get(model_name, {})
    model_params = {k: v for k, v in model_config.items() if k != "name"}
    model = create_model(model_name, **model_params)

    if model_name == "lightgbm":
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    # Save feature schema (same for all models)
    _save_feature_schema(feature_schema_path)

    # Save model artifact (fix #4: joblib for LightGBM)
    models_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "lightgbm":
        model_path = models_dir / f"{model_name}.joblib"
        model.save(model_path)
        logger.info("model_saved", path=str(model_path))

    # MLflow logging
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(model.get_params())
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))

        val_preds = model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        mlflow.log_metric("val_mae", val_mae)
        logger.info("val_mae", model=model_name, mae=val_mae)

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "val_mae": val_mae,
            "model": model,
        }


def _save_feature_schema(path: Path) -> None:
    schema = {
        "columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "categorical": ["zone_id"],
        "numerical": [c for c in FEATURE_COLUMNS if c != "zone_id"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2))


def train_all(
    features_path: Path,
    config: dict,
    models_dir: Path,
    feature_schema_path: Path,
) -> dict[str, Any]:
    """Train all models defined in config."""
    results = {}
    for model_name in config["models"]:
        result = train_model(
            features_path=features_path,
            config=config,
            model_name=model_name,
            models_dir=models_dir,
            feature_schema_path=feature_schema_path,
        )
        results[model_name] = result
    return results
```

**Step 2: Write scripts/train.py**

```python
"""Script entrypoint for model training."""

from pathlib import Path

import yaml

from demandops.training.train import train_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    results = train_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
        feature_schema_path=Path(config["artifacts"]["feature_schema_path"]),
    )
    for name, info in results.items():
        print(f"{name}: val_mae={info['val_mae']:.4f}, run_id={info['run_id']}")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add demandops/training/train.py scripts/train.py
git commit -m "feat: add training pipeline with MLflow tracking and joblib save"
```

---

### Task 14: Evaluation

**Files:**
- Create: `demandops/training/evaluate.py`
- Create: `scripts/evaluate.py`

**Step 1: Write demandops/training/evaluate.py**

Key fix #7: use `predict_raw()` for accurate clipping stats.

```python
"""Evaluation: compute metrics on test set, produce report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from demandops.data.splits import split_from_config
from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN

logger = structlog.get_logger()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error. Handles zeros."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(
        100.0 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask])
    )


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a single model on the test set."""
    start = time.perf_counter()
    preds = model.predict(X_test)
    latency_ms = (time.perf_counter() - start) * 1000

    # Clipping stats (fix #7: use predict_raw for accurate count)
    n_clipped = 0
    if model_name == "lightgbm" and hasattr(model, "predict_raw"):
        raw_preds = model.predict_raw(X_test)
        n_clipped = int(np.sum(raw_preds < 0))

    metrics = {
        "model_name": model_name,
        "mae": mae(y_test, preds),
        "rmse": rmse(y_test, preds),
        "smape": smape(y_test, preds),
        "latency_ms": latency_ms,
        "n_predictions": len(preds),
        "n_clipped_to_zero": n_clipped,
        "pct_clipped": round(100 * n_clipped / len(preds), 2) if len(preds) > 0 else 0,
    }

    logger.info("evaluation_complete", **metrics)
    return metrics


def evaluate_all(
    features_path: Path,
    config: dict,
    trained_models: dict[str, Any],
    reports_dir: Path,
    zone_universe_path: Path,
) -> dict:
    """Evaluate all trained models on test set. Save report."""
    df = pl.read_parquet(features_path)
    _, _, test = split_from_config(df, config)

    X_test = test.select(FEATURE_COLUMNS).to_numpy()
    y_test = test[TARGET_COLUMN].to_numpy().astype(float)

    logger.info("test_set_size", rows=len(test))

    results = {}
    for model_name, model_info in trained_models.items():
        results[model_name] = evaluate_model(
            model_info["model"], X_test, y_test, model_name
        )

    # Feature importance
    lgbm_info = trained_models.get("lightgbm")
    feature_importance = None
    if lgbm_info and hasattr(lgbm_info["model"], "feature_importances"):
        importances = lgbm_info["model"].feature_importances
        if importances is not None:
            feature_importance = sorted(
                zip(FEATURE_COLUMNS, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

    per_zone = _per_zone_analysis(test, trained_models)
    edge_cases = _edge_case_analysis(test, trained_models)

    report = {
        "model_comparison": results,
        "feature_importance": feature_importance,
        "per_zone_top5": per_zone,
        "edge_cases": edge_cases,
        "test_rows": len(test),
        "config_snapshot": config,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "eval_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_saved", path=str(report_path))

    return report


def _per_zone_analysis(
    test_df: pl.DataFrame, trained_models: dict
) -> list[dict]:
    """Top 5 hardest zones by LightGBM MAE."""
    lgbm = trained_models.get("lightgbm")
    if not lgbm:
        return []

    model = lgbm["model"]
    X = test_df.select(FEATURE_COLUMNS).to_numpy()
    preds = model.predict(X)

    test_with_preds = test_df.with_columns(pl.Series("pred", preds))
    zone_mae = (
        test_with_preds
        .with_columns((pl.col("pred") - pl.col("trip_count")).abs().alias("abs_error"))
        .group_by("zone_id")
        .agg(
            pl.col("abs_error").mean().alias("mae"),
            pl.col("trip_count").mean().alias("mean_demand"),
        )
        .sort("mae", descending=True)
        .head(5)
    )
    return zone_mae.to_dicts()


def _edge_case_analysis(
    test_df: pl.DataFrame, trained_models: dict
) -> dict:
    """Edge-case segment analysis."""
    results = {}

    zone_means = test_df.group_by("zone_id").agg(
        pl.col("trip_count").mean().alias("mean_demand")
    )
    p10 = zone_means["mean_demand"].quantile(0.1)
    p90 = zone_means["mean_demand"].quantile(0.9)

    sparse_zones = set(
        zone_means.filter(pl.col("mean_demand") < p10)["zone_id"].to_list()
    )
    dense_zones = set(
        zone_means.filter(pl.col("mean_demand") > p90)["zone_id"].to_list()
    )

    segments = {
        "sparse_zones": test_df.filter(pl.col("zone_id").is_in(sparse_zones)),
        "dense_zones": test_df.filter(pl.col("zone_id").is_in(dense_zones)),
        "late_night": test_df.filter(pl.col("hour_of_day").is_between(0, 5)),
        "peak_hours": test_df.filter(
            pl.col("hour_of_day").is_in([7, 8, 9, 17, 18, 19])
        ),
        "weekend": test_df.filter(pl.col("is_weekend") == 1),
        "weekday": test_df.filter(pl.col("is_weekend") == 0),
        "zero_demand": test_df.filter(pl.col("trip_count") == 0),
    }

    for seg_name, seg_df in segments.items():
        if len(seg_df) == 0:
            continue
        X_seg = seg_df.select(FEATURE_COLUMNS).to_numpy()
        y_seg = seg_df["trip_count"].to_numpy().astype(float)

        seg_result = {"n_rows": len(seg_df)}
        for model_name, model_info in trained_models.items():
            preds = model_info["model"].predict(X_seg)
            seg_result[f"{model_name}_mae"] = mae(y_seg, preds)

        results[seg_name] = seg_result

    return results
```

**Step 2: Write scripts/evaluate.py**

```python
"""Script entrypoint for evaluation."""

from pathlib import Path

import yaml

from demandops.training.evaluate import evaluate_all
from demandops.training.train import train_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())

    trained = train_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
        feature_schema_path=Path(config["artifacts"]["feature_schema_path"]),
    )

    report = evaluate_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        trained_models=trained,
        reports_dir=Path(config["artifacts"]["reports_dir"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
    )

    print("\n=== Model Comparison ===")
    for name, metrics in report["model_comparison"].items():
        print(f"{name}: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
              f"sMAPE={metrics['smape']:.2f}%")


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add demandops/training/evaluate.py scripts/evaluate.py
git commit -m "feat: add evaluation with accurate clipping stats via predict_raw"
```

---

### Task 15: Full Test Suite Check

```bash
make test
```

Expected: ALL PASS. This is the Day 2 gate.

---

## Day 3: Serving + FeatureService (Tasks 16–23)

**Done when:** `curl POST /predict` works. Invalid zone → 422. December timestamp → 422. `/health` shows n_supported_zones. `/metrics` returns Prometheus format. All serving tests pass.

---

### Task 16: Feature Service

**Files:**
- Create: `demandops/serving/feature_service.py`
- Create: `tests/test_feature_service.py`

**Step 1: Write tests/test_feature_service.py**

```python
"""Tests for FeatureService."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from demandops.serving.feature_service import FeatureService


class TestFeatureService:

    def test_valid_zone_and_timestamp(self, feature_service: FeatureService) -> None:
        result = feature_service.get_features(1, datetime(2024, 2, 1, 12, 0))
        assert result.supported
        assert result.features is not None
        assert len(result.features) == 9

    def test_zone_not_in_universe(self, feature_service: FeatureService) -> None:
        result = feature_service.get_features(999, datetime(2024, 2, 1, 12, 0))
        assert not result.supported
        assert result.features is None
        assert any("zone universe" in w for w in result.warnings)

    def test_zone_in_range_but_not_in_universe(
        self, feature_service: FeatureService
    ) -> None:
        """Zone 100 is in 1-263 but not in test universe [1,2,3]."""
        result = feature_service.get_features(100, datetime(2024, 2, 1, 12, 0))
        assert not result.supported

    def test_timestamp_before_supported_start(
        self, feature_service: FeatureService
    ) -> None:
        result = feature_service.get_features(1, datetime(2023, 12, 15, 12, 0))
        assert not result.supported

    def test_timestamp_at_supported_end_exclusive(
        self, feature_service: FeatureService
    ) -> None:
        result = feature_service.get_features(1, feature_service.supported_end)
        assert not result.supported

    def test_supported_start(self, feature_service: FeatureService) -> None:
        assert feature_service.supported_start == datetime(2024, 1, 1)

    def test_supported_end(self, feature_service: FeatureService) -> None:
        assert feature_service.supported_end == datetime(2024, 3, 1)

    def test_n_supported_zones(self, feature_service: FeatureService) -> None:
        assert feature_service.n_supported_zones == 3

    def test_feature_order_matches_schema(
        self, feature_service: FeatureService
    ) -> None:
        from demandops.features import FEATURE_COLUMNS
        result = feature_service.get_features(1, datetime(2024, 2, 1, 12, 0))
        assert result.supported
        assert list(result.features.keys()) == FEATURE_COLUMNS

    def test_weekday_matches_python_convention(
        self, feature_service: FeatureService
    ) -> None:
        """FeatureService uses datetime.weekday() → 0=Mon.
        Monday 2024-01-01 should have day_of_week=0."""
        result = feature_service.get_features(1, datetime(2024, 1, 1, 12, 0))
        assert result.supported
        assert result.features["day_of_week"] == 0  # Monday

    def test_last_supported_hour(self, feature_service: FeatureService) -> None:
        last_hour = feature_service.supported_end - timedelta(hours=1)
        result = feature_service.get_features(1, last_hour)
        assert result.supported
```

**Step 2: Run tests (expect failure)**

```bash
pytest tests/test_feature_service.py -v
```

**Step 3: Write demandops/serving/feature_service.py**

```python
"""FeatureService: reconstruct lag features at request time from dense history.

Uses Python datetime.weekday() (0=Mon, 6=Sun) for consistency with
the training pipeline which normalizes Polars weekday to the same convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

from demandops.features import FEATURE_COLUMNS

logger = structlog.get_logger()


@dataclass
class FeatureResult:
    features: dict | None
    supported: bool
    warnings: list[str] = field(default_factory=list)


class FeatureService:
    """Serves features for prediction requests.

    Loads the dense hourly history grid and reconstructs lag features
    at request time, ensuring train-serve parity.
    """

    def __init__(
        self,
        history_path: Path,
        schema_path: Path,
        zone_universe_path: Path,
        config: dict,
    ) -> None:
        self.history = pl.read_parquet(history_path)
        self.schema = json.loads(Path(schema_path).read_text())
        zone_data = json.loads(Path(zone_universe_path).read_text())
        self.zone_universe: set[int] = set(zone_data["zone_ids"])

        self._min_history_ts: datetime = self.history["hour_ts"].min()
        self._max_history_ts: datetime = self.history["hour_ts"].max()
        self._train_start = datetime.fromisoformat(config["split"]["train_start"])

        # Build lookup: (zone_id, hour_ts) → trip_count
        self._lookup: dict[tuple[int, datetime], int] = {}
        self._zone_names: dict[int, str] = {}
        for row in self.history.iter_rows(named=True):
            key = (row["zone_id"], row["hour_ts"])
            self._lookup[key] = row["trip_count"]
            if row["zone_id"] not in self._zone_names:
                self._zone_names[row["zone_id"]] = row["zone_name"]

        logger.info(
            "feature_service_loaded",
            history_rows=len(self.history),
            n_zones=len(self.zone_universe),
            supported_start=str(self.supported_start),
            supported_end=str(self.supported_end),
        )

    @property
    def supported_start(self) -> datetime:
        return self._train_start

    @property
    def supported_end(self) -> datetime:
        return self._max_history_ts + timedelta(hours=1)

    @property
    def n_supported_zones(self) -> int:
        return len(self.zone_universe)

    def get_zone_name(self, zone_id: int) -> str:
        return self._zone_names.get(zone_id, f"Unknown Zone {zone_id}")

    def get_features(self, zone_id: int, hour_ts: datetime) -> FeatureResult:
        # Normalize to naive datetime for consistent lookup (fix #15)
        # Pydantic may parse "2024-02-01T12:00:00Z" as timezone-aware
        if hour_ts.tzinfo is not None:
            hour_ts = hour_ts.replace(tzinfo=None)

        warnings: list[str] = []

        if zone_id not in self.zone_universe:
            return FeatureResult(
                features=None, supported=False,
                warnings=[f"zone_id {zone_id} not in supported zone universe"],
            )

        if hour_ts < self.supported_start or hour_ts >= self.supported_end:
            return FeatureResult(
                features=None, supported=False,
                warnings=[
                    f"hour_ts {hour_ts.isoformat()} outside supported range "
                    f"[{self.supported_start.isoformat()}, "
                    f"{self.supported_end.isoformat()})"
                ],
            )

        # Temporal features (Python weekday: 0=Mon, 6=Sun)
        day_of_week = hour_ts.weekday()

        # Lag features from dense history
        lag_1h = self._get_trip_count(zone_id, hour_ts - timedelta(hours=1))
        lag_24h = self._get_trip_count(zone_id, hour_ts - timedelta(hours=24))
        lag_168h = self._get_trip_count(zone_id, hour_ts - timedelta(hours=168))

        # Rolling mean 24h: mean of trip_count at hours [t-24, t-1]
        rolling_vals = []
        for offset in range(1, 25):
            val = self._get_trip_count(zone_id, hour_ts - timedelta(hours=offset))
            if val is not None:
                rolling_vals.append(val)
        rolling_mean_24h = (
            sum(rolling_vals) / len(rolling_vals) if rolling_vals else 0.0
        )

        # Build features dict in FEATURE_COLUMNS order
        features = {
            "hour_of_day": hour_ts.hour,
            "day_of_week": day_of_week,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "month": hour_ts.month,
            "zone_id": zone_id,
            "lag_1h": float(lag_1h) if lag_1h is not None else 0.0,
            "lag_24h": float(lag_24h) if lag_24h is not None else 0.0,
            "lag_168h": float(lag_168h) if lag_168h is not None else 0.0,
            "rolling_mean_24h": rolling_mean_24h,
        }

        # Verify key order matches FEATURE_COLUMNS
        assert list(features.keys()) == FEATURE_COLUMNS

        return FeatureResult(features=features, supported=True, warnings=warnings)

    def _get_trip_count(self, zone_id: int, hour_ts: datetime) -> int | None:
        return self._lookup.get((zone_id, hour_ts))
```

**Step 4: Run tests**

```bash
pytest tests/test_feature_service.py -v
```

Expected: ALL PASS.

**Step 5: Commit**

```bash
git add demandops/serving/feature_service.py tests/test_feature_service.py
git commit -m "feat: add FeatureService with train-serve parity on weekday convention"
```

---

### Task 17: Serving Schemas

**Files:**
- Create: `demandops/serving/schemas.py`

**Step 1: Write demandops/serving/schemas.py**

```python
"""Pydantic v2 schemas for the serving API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    zone_id: int = Field(ge=1, le=263)
    hour_ts: datetime


class PredictionMetadata(BaseModel):
    latency_ms: float
    request_id: str
    features_used: dict
    input_warnings: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    zone_id: int
    zone_name: str
    hour_ts: datetime
    predicted_count: float = Field(ge=0.0)
    model_name: str
    metadata: PredictionMetadata


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: str
    history_loaded: bool
    supported_start: datetime
    supported_end: datetime
    n_supported_zones: int
    history_rows: int
    uptime_seconds: float


class ErrorDetail(BaseModel):
    detail: str
    supported_start: datetime | None = None
    supported_end: datetime | None = None
    n_supported_zones: int | None = None
```

**Step 2: Commit**

```bash
git add demandops/serving/schemas.py
git commit -m "feat: add Pydantic serving schemas"
```

---

### Task 18: Prometheus Metrics

**Files:**
- Create: `demandops/serving/metrics.py`

**Step 1: Write demandops/serving/metrics.py**

```python
"""Prometheus metric definitions using prometheus-client."""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "demandops_requests_total",
    "Total HTTP requests",
    ["endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "demandops_predictions_total",
    "Total successful predictions",
)

REJECTION_COUNT = Counter(
    "demandops_rejections_total",
    "Total rejected requests (unsupported zone or timestamp)",
    ["reason"],
)

ERROR_COUNT = Counter(
    "demandops_errors_total",
    "Total internal errors",
)

REQUEST_LATENCY = Histogram(
    "demandops_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_VALUE = Histogram(
    "demandops_prediction_value",
    "Distribution of predicted trip counts",
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

MODEL_LOADED = Gauge(
    "demandops_model_loaded",
    "Whether the model is loaded (1=yes, 0=no)",
)

HISTORY_LOADED = Gauge(
    "demandops_history_loaded",
    "Whether the history table is loaded (1=yes, 0=no)",
)
```

**Step 2: Commit**

```bash
git add demandops/serving/metrics.py
git commit -m "feat: add Prometheus metric definitions"
```

---

### Task 19: Routes + App + Middleware

**Files:**
- Create: `demandops/serving/routes.py`
- Create: `demandops/serving/app.py`
- Create: `demandops/serving/middleware.py`

**Step 1: Write demandops/serving/routes.py (fix #8: use app.state, not globals)**

```python
"""API routes: /predict, /health, /metrics."""

from __future__ import annotations

import time
import uuid

import numpy as np
import structlog
from fastapi import APIRouter, FastAPI, HTTPException, Request
from prometheus_client import generate_latest
from starlette.responses import Response

from demandops.serving.metrics import (
    ERROR_COUNT,
    HISTORY_LOADED,
    MODEL_LOADED,
    PREDICTION_COUNT,
    PREDICTION_VALUE,
    REJECTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from demandops.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionMetadata,
)

logger = structlog.get_logger()

router = APIRouter()


def configure(
    app: FastAPI, feature_service, model, model_name: str, start_time: float
):
    """Store dependencies on app.state (fix #8: no module-level globals)."""
    app.state.feature_service = feature_service
    app.state.model = model
    app.state.model_name = model_name
    app.state.start_time = start_time


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    svc = request.app.state.feature_service
    model = request.app.state.model
    model_name = request.app.state.model_name

    try:
        result = svc.get_features(body.zone_id, body.hour_ts)

        if not result.supported:
            reason = (
                "unsupported_zone"
                if result.warnings and "zone universe" in result.warnings[0]
                else "unsupported_timestamp"
            )
            REJECTION_COUNT.labels(reason=reason).inc()
            REQUEST_COUNT.labels(endpoint="/predict", status="422").inc()
            REQUEST_LATENCY.labels(endpoint="/predict").observe(
                time.perf_counter() - start
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "detail": result.warnings[0] if result.warnings else "Unsupported",
                    "supported_start": svc.supported_start.isoformat(),
                    "supported_end": svc.supported_end.isoformat(),
                    "n_supported_zones": svc.n_supported_zones,
                },
            )

        features = result.features
        X = np.array([[features[col] for col in features]], dtype=float)

        predicted_count = float(model.predict(X)[0])
        latency_ms = (time.perf_counter() - start) * 1000

        PREDICTION_COUNT.inc()
        PREDICTION_VALUE.observe(predicted_count)
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(
            time.perf_counter() - start
        )

        return PredictResponse(
            zone_id=body.zone_id,
            zone_name=svc.get_zone_name(body.zone_id),
            hour_ts=body.hour_ts,
            predicted_count=predicted_count,
            model_name=model_name,
            metadata=PredictionMetadata(
                latency_ms=latency_ms,
                request_id=request_id,
                features_used=features,
                input_warnings=result.warnings,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(
            time.perf_counter() - start
        )
        logger.error("prediction_error", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    svc = request.app.state.feature_service
    model = request.app.state.model
    model_name = request.app.state.model_name
    start_time = request.app.state.start_time

    model_loaded = model is not None
    history_loaded = svc is not None

    MODEL_LOADED.set(1 if model_loaded else 0)
    HISTORY_LOADED.set(1 if history_loaded else 0)
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()

    return HealthResponse(
        status="healthy" if model_loaded and history_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name or "none",
        history_loaded=history_loaded,
        supported_start=svc.supported_start if history_loaded else None,
        supported_end=svc.supported_end if history_loaded else None,
        n_supported_zones=svc.n_supported_zones if history_loaded else 0,
        history_rows=len(svc.history) if history_loaded else 0,
        uptime_seconds=time.time() - start_time if start_time else 0,
    )


@router.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
```

**Step 2: Write demandops/serving/middleware.py**

```python
"""Request logging middleware."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=request_id,
        )
        return response
```

**Step 3: Write demandops/serving/app.py**

```python
"""FastAPI application factory."""

from __future__ import annotations

import time
from pathlib import Path

import structlog
import yaml
from fastapi import FastAPI

from demandops.models.registry import create_model
from demandops.serving.feature_service import FeatureService
from demandops.serving.middleware import RequestLoggingMiddleware
from demandops.serving.routes import configure, router

logger = structlog.get_logger()


def create_app(config_path: str = "configs/default.yaml") -> FastAPI:
    config = yaml.safe_load(Path(config_path).read_text())
    serving_cfg = config["serving"]

    app = FastAPI(
        title="demandops-lite",
        description="Hourly taxi demand prediction API",
        version="0.1.0",
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        start_time = time.time()

        feature_service = FeatureService(
            history_path=Path(serving_cfg["history_path"]),
            schema_path=Path(serving_cfg["feature_schema_path"]),
            zone_universe_path=Path(serving_cfg["zone_universe_path"]),
            config=config,
        )

        # Load model (fix #4: joblib for LightGBM)
        model_name = serving_cfg["model_name"]
        model_config = config["models"].get(model_name, {})
        model_params = {k: v for k, v in model_config.items() if k != "name"}
        model = create_model(model_name, **model_params)

        if model_name == "lightgbm":
            model_path = Path(config["artifacts"]["models_dir"]) / f"{model_name}.joblib"
            if model_path.exists():
                model.load(model_path)
                logger.info("model_loaded", path=str(model_path))
            else:
                logger.warning("model_file_not_found", path=str(model_path))

        configure(app, feature_service, model, model_name, start_time)
        logger.info("app_started", model=model_name)

    return app


app = create_app()
```

**Step 4: Commit**

```bash
git add demandops/serving/routes.py demandops/serving/middleware.py demandops/serving/app.py
git commit -m "feat: add FastAPI routes (app.state DI), middleware, app factory"
```

---

### Task 20: Monitoring Checks

**Files:**
- Create: `demandops/monitoring/checks.py`

**Step 1: Write demandops/monitoring/checks.py**

```python
"""Input validation and monitoring checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    warnings: list[str]


def check_sparse_zone(
    zone_id: int,
    zone_mean_demands: dict[int, float],
    threshold_percentile: float = 10.0,
) -> list[str]:
    """Check if a zone is sparse (below P10 mean demand)."""
    warnings = []
    if zone_id in zone_mean_demands:
        values = sorted(zone_mean_demands.values())
        if values:
            idx = int(len(values) * threshold_percentile / 100)
            p10 = values[min(idx, len(values) - 1)]
            if zone_mean_demands[zone_id] < p10:
                warnings.append(
                    f"zone_id {zone_id} is a sparse zone "
                    f"(mean demand {zone_mean_demands[zone_id]:.1f} < P10={p10:.1f})"
                )
    return warnings


def check_extreme_prediction(
    predicted_count: float,
    zone_id: int,
    zone_max_demands: dict[int, float],
    threshold_multiplier: float = 5.0,
) -> list[str]:
    """Flag predictions unusually high for the zone."""
    warnings = []
    if zone_id in zone_max_demands:
        max_seen = zone_max_demands[zone_id]
        if max_seen > 0 and predicted_count > max_seen * threshold_multiplier:
            warnings.append(
                f"Prediction {predicted_count:.1f} exceeds "
                f"{threshold_multiplier}x max historical ({max_seen:.1f}) "
                f"for zone {zone_id}"
            )
    return warnings
```

**Step 2: Commit**

```bash
git add demandops/monitoring/checks.py
git commit -m "feat: add monitoring checks (sparse zone, extreme prediction)"
```

---

### Task 21: tests/test_serving.py

**Files:**
- Create: `tests/test_serving.py`

**Step 1: Write tests/test_serving.py**

Note: `test_client` fixture is already in `conftest.py` (fix #9).

```python
"""Tests for the serving API endpoints."""

from __future__ import annotations


class TestPredictEndpoint:

    def test_valid_request(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["zone_id"] == 1
        assert data["predicted_count"] == 42.5
        assert data["model_name"] == "lightgbm"

    def test_unsupported_zone(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 100,
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_december_timestamp(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2023-12-15T12:00:00",
        })
        assert resp.status_code == 422

    def test_response_has_metadata(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-02-01T12:00:00",
        })
        data = resp.json()
        assert "metadata" in data
        assert "latency_ms" in data["metadata"]
        assert "request_id" in data["metadata"]
        assert "features_used" in data["metadata"]


class TestHealthEndpoint:

    def test_health_returns_200(self, test_client) -> None:
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_includes_n_supported_zones(self, test_client) -> None:
        data = test_client.get("/health").json()
        assert "n_supported_zones" in data
        assert data["n_supported_zones"] == 3

    def test_health_includes_supported_range(self, test_client) -> None:
        data = test_client.get("/health").json()
        assert "supported_start" in data
        assert "supported_end" in data


class TestMetricsEndpoint:

    def test_metrics_returns_text_plain(self, test_client) -> None:
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_prometheus_metrics(self, test_client) -> None:
        test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-02-01T12:00:00",
        })
        body = test_client.get("/metrics").text
        assert "demandops_requests_total" in body
        assert "demandops_request_latency_seconds" in body
```

**Step 2: Run tests**

```bash
pytest tests/test_serving.py -v
```

**Step 3: Commit**

```bash
git add tests/test_serving.py
git commit -m "test: add serving API tests"
```

---

### Task 22: tests/test_edge_cases.py

**Files:**
- Create: `tests/test_edge_cases.py`

**Step 1: Write tests/test_edge_cases.py**

```python
"""Edge-case tests for the serving layer."""

from __future__ import annotations


class TestEdgeCases:

    def test_zone_id_zero(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 0,
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_zone_id_264(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 264,
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_missing_zone_id(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "hour_ts": "2024-02-01T12:00:00",
        })
        assert resp.status_code == 422

    def test_missing_hour_ts(self, test_client) -> None:
        resp = test_client.post("/predict", json={"zone_id": 1})
        assert resp.status_code == 422

    def test_invalid_hour_ts(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "not-a-date",
        })
        assert resp.status_code == 422

    def test_feb_29_2024_valid(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-02-29T12:00:00",
        })
        assert resp.status_code == 200

    def test_boundary_supported_start(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-01-01T00:00:00",
        })
        assert resp.status_code == 200

    def test_boundary_supported_end_exclusive(self, test_client) -> None:
        resp = test_client.post("/predict", json={
            "zone_id": 1,
            "hour_ts": "2024-03-01T00:00:00",
        })
        assert resp.status_code == 422
```

**Step 2: Run all tests**

```bash
make test
```

Expected: ALL PASS.

**Step 3: Commit**

```bash
git add tests/test_edge_cases.py
git commit -m "test: add edge-case tests"
```

---

### Task 23: Full Day 3 Gate Check

```bash
make test
```

ALL PASS = Day 3 complete.

---

## Day 4: Benchmark + CI + Docker (Tasks 24–27)

**Done when:** Benchmark report has real numbers. Docker-compose up works. CI green.

---

### Task 24: Benchmark Script

**Files:**
- Create: `scripts/benchmark.py`

**Step 1: Write scripts/benchmark.py**

```python
"""Run full benchmark: train → evaluate → markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from demandops.training.evaluate import evaluate_all
from demandops.training.train import train_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())

    print("Training all models...")
    trained = train_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
        feature_schema_path=Path(config["artifacts"]["feature_schema_path"]),
    )

    print("Evaluating on test set...")
    report = evaluate_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        trained_models=trained,
        reports_dir=Path(config["artifacts"]["reports_dir"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
    )

    _generate_markdown_report(report, config)


def _generate_markdown_report(report: dict, config: dict) -> None:
    zone_universe = json.loads(
        Path(config["artifacts"]["zone_universe_path"]).read_text()
    )

    lines = [
        "## Benchmark Results — NYC Taxi Demand Prediction\n",
        f"**Dataset:** NYC TLC Yellow Taxi, Jan–Feb 2024 (Dec 2023 for warm-up)",
        f"**Target:** Hourly trip count per pickup zone",
        f"**Zones:** {zone_universe['n_zones']} (from zone_universe.json)",
        f"**Train:** [2024-01-01, 2024-02-01) | **Val:** [2024-02-01, 2024-02-15) | **Test:** [2024-02-15, 2024-03-01)",
        f"**Features:** 9 (temporal + lag)",
        "",
    ]

    lgbm = report["model_comparison"].get("lightgbm", {})
    if lgbm:
        lines.append(
            f"**Negative prediction handling:** LightGBM predictions clipped to zero "
            f"({lgbm.get('n_clipped_to_zero', 0)} predictions, "
            f"{lgbm.get('pct_clipped', 0):.1f}%)"
        )
        lines.append("")

    lines.append("### Model Comparison\n")
    lines.append("| Model | MAE | RMSE | sMAPE | Latency (ms) |")
    lines.append("|-------|-----|------|-------|-------------|")

    for name in ["slot_mean", "seasonal_naive", "lightgbm"]:
        m = report["model_comparison"].get(name, {})
        lines.append(
            f"| {name} | {m.get('mae', 0):.2f} | {m.get('rmse', 0):.2f} | "
            f"{m.get('smape', 0):.2f}% | {m.get('latency_ms', 0):.1f} |"
        )

    # Delta rows
    sm = report["model_comparison"].get("slot_mean", {})
    sn = report["model_comparison"].get("seasonal_naive", {})
    lg = report["model_comparison"].get("lightgbm", {})

    if sm and lg and sm.get("mae", 0) > 0:
        d = (lg["mae"] - sm["mae"]) / sm["mae"] * 100
        lines.append(f"| **vs Slot Mean** | {d:+.1f}% | — | — | — |")
    if sn and lg and sn.get("mae", 0) > 0:
        d = (lg["mae"] - sn["mae"]) / sn["mae"] * 100
        lines.append(f"| **vs Seasonal Naive** | {d:+.1f}% | — | — | — |")

    if report.get("feature_importance"):
        lines.append("\n### Feature Importance (LightGBM, top 10)\n")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for i, (feat, imp) in enumerate(report["feature_importance"], 1):
            lines.append(f"| {i} | {feat} | {imp:.4f} |")

    if report.get("edge_cases"):
        lines.append("\n### Edge-Case Analysis\n")
        lines.append("| Segment | N rows | Slot Mean MAE | LightGBM MAE |")
        lines.append("|---------|--------|-------------|-------------|")
        for seg, data in report["edge_cases"].items():
            sm_m = data.get("slot_mean_mae", "—")
            lg_m = data.get("lightgbm_mae", "—")
            if isinstance(sm_m, float):
                sm_m = f"{sm_m:.2f}"
            if isinstance(lg_m, float):
                lg_m = f"{lg_m:.2f}"
            lines.append(f"| {seg} | {data['n_rows']} | {sm_m} | {lg_m} |")

    report_path = Path("docs/benchmark_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nBenchmark report saved to {report_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/benchmark.py
git commit -m "feat: add benchmark script with markdown report"
```

---

### Task 25: CI Workflow

**Files:**
- Create: `.github/workflows/ci.yaml`

**Step 1: Write .github/workflows/ci.yaml**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint
        run: ruff check demandops/ tests/

      - name: Test
        run: pytest tests/ -v --tb=short
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yaml
git commit -m "ci: add GitHub Actions workflow"
```

---

### Task 26: Docker (fix #14)

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yaml`

**Step 1: Write docker/Dockerfile (fix #14: proper layer caching)**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy just pyproject.toml first for dependency layer caching
COPY pyproject.toml .

# Create minimal package structure so pip install -e works
RUN mkdir -p demandops && touch demandops/__init__.py

# Install dependencies (cached unless pyproject.toml changes)
RUN pip install --no-cache-dir .

# Now copy actual source
COPY demandops/ demandops/
COPY configs/ configs/

EXPOSE 8001

CMD ["uvicorn", "demandops.serving.app:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Step 2: Write docker/docker-compose.yaml**

```yaml
version: "3.8"

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8001:8001"
    volumes:
      - ../data/processed:/app/data/processed:ro
      - ../artifacts:/app/artifacts:ro
    environment:
      - PYTHONUNBUFFERED=1
```

**Step 3: Commit**

```bash
git add docker/Dockerfile docker/docker-compose.yaml
git commit -m "feat: add Docker with proper layer caching"
```

---

### Task 27: Day 4 Gate Verification

Run in order:

```bash
# 1. Full pipeline on real data
make pipeline

# 2. Verify benchmark report
cat docs/benchmark_report.md

# 3. Docker
cd docker && docker-compose up --build -d
curl http://localhost:8001/health
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"zone_id": 161, "hour_ts": "2024-02-20T14:00:00"}'
curl http://localhost:8001/metrics
docker-compose down && cd ..

# 4. Tests
make test
```

All green = Day 4 gate passed.

---

## Day 5: Polish (Tasks 28–29)

### Task 28: README.md

Write following spec section 12 structure. Fill in real benchmark numbers from `docs/benchmark_report.md`.

### Task 29: DECISIONS.md

Copy from spec section 11 verbatim.

---

## Implementation Notes

### Weekday Convention (fix #1/#3)

**Single source of truth:** `day_of_week` is 0=Mon, 6=Sun everywhere.

| Module | API | Raw return | Our normalization |
|--------|-----|-----------|-------------------|
| Polars `dt.weekday()` | 1=Mon, 7=Sun | Subtract 1 | `dt.weekday() - 1` |
| Python `datetime.weekday()` | 0=Mon, 6=Sun | Already correct | No change |
| `is_weekend` | `day_of_week >= 5` | Sat=5, Sun=6 | Same in both |

Test: `test_weekday_matches_python_convention` verifies 2024-01-01 (Monday) → 0.

### Feature Column Order

Canonical order defined in `demandops/features.py:FEATURE_COLUMNS`. Used by:
- `training/train.py` → extract from DataFrame
- `training/evaluate.py` → extract from DataFrame
- `serving/feature_service.py` → build feature dict
- `serving/routes.py` → build numpy array
- `models/baselines.py` → column index constants (via `IDX_*`)
- `feature_schema.json` → persisted artifact

### Model Serialization (fix #4)

LightGBM models saved/loaded via `joblib`. The sklearn wrapper's internal `_Booster` attribute is not a reliable serialization boundary. `joblib` preserves the full `LGBMRegressor` state including fitted attributes (`best_iteration_`, `feature_importances_`, etc.).

### Rolling Mean (fix #2)

Uses `group_by("zone_id", maintain_order=True).map_groups()` with `shift(1).rolling_mean()` applied per group. This avoids the version-dependent behavior of `rolling_mean().over()`, where the rolling window function operates on physical row positions and `.over()` partitioning doesn't guarantee correct boundaries. The `map_groups` approach is unambiguous: each group is an isolated sorted DataFrame. Smoke-test in Task 6 verifies correctness before proceeding.

### Prometheus in Tests

If Prometheus collectors conflict between tests (multiple `TestClient` instances), reset via:

```python
from prometheus_client import REGISTRY
# In conftest.py or fixture
for name in list(REGISTRY._names_to_collectors.keys()):
    try:
        REGISTRY.unregister(REGISTRY._names_to_collectors[name])
    except Exception:
        pass
```
