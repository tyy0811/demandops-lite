# Citibike Adapter -- Implementation Plan

## Data source

Public S3 bucket: `https://s3.amazonaws.com/tripdata/`. Monthly zip files
containing CSVs. For Dec 2023 through Feb 2024:

- `202312-citibike-tripdata.csv.zip`
- `202401-citibike-tripdata.csv.zip`
- `202402-citibike-tripdata.csv.zip`

The 2020+ schema columns: `ride_id`, `rideable_type`, `started_at`,
`ended_at`, `start_station_name`, `start_station_id`, `end_station_name`,
`end_station_id`, `start_lat`, `start_lng`, `end_lat`, `end_lng`,
`member_casual`.

**Quirk 1 -- multi-CSV zips:** months with >1M trips split data across
multiple CSVs within the same zip. Dec--Feb NYC winter should be under
1M/month, but the adapter must handle multiple CSVs regardless.

**Quirk 2 -- string station IDs:** `start_station_id` contains strings like
`"6215.01"` or letter-prefixed IDs. Handled by filtering (see design decision
below).

## Adapter mapping

| Citibike CSV column                  | Common schema |
|--------------------------------------|---------------|
| `start_station_id` (filtered, cast)  | `zone_id`     |
| `start_station_name`                 | `zone_name`   |
| `started_at` (truncated to hour)     | `hour_ts`     |
| `COUNT(*)`                           | `trip_count`  |
| _(null)_                             | `avg_fare`    |
| _(null)_                             | `avg_distance`|

## Design decision: string station IDs

The common schema has `zone_id` as `int` (Pandera: `pa.Field(ge=1)`).
Citibike's `start_station_id` contains non-numeric strings.

**Option A -- Hash to int.** Deterministic, fits schema, loses readability.

**Option B -- Change schema to string.** Correct but touches features.py,
Pandera schemas, serving schemas, all existing tests. High-risk refactor.

**Option C -- Filter to numeric-only station IDs (chosen).** Drop rows where
`start_station_id` is null or non-numeric. ~2--3% data loss. Zero schema
changes. Same YAGNI reasoning as keeping `zone_id` instead of renaming to
`entity_id`. Document in DECISIONS.md entry #21.

## Files to create

| File | Purpose |
|------|---------|
| `demandops/data/adapters/citibike.py` | `CitibikeAdapter` class |
| `configs/citibike.yaml` | Citibike pipeline config |
| `tests/test_citibike_adapter.py` | Mock CSV tests (no network) |

## Files to modify

| File | Change |
|------|--------|
| `demandops/data/adapters/__init__.py` | Register `"citibike"` in `ADAPTER_REGISTRY` |
| `README.md` | Triple-dataset benchmark table |
| `DECISIONS.md` | Entry #21: Citibike adapter + station ID filtering rationale |

## Files unchanged

Everything else: `features.py`, `train.py`, `evaluate.py`, `serving/*`,
`schemas.py`, existing tests, CI workflow. That's the point of
`DatasetAdapter`.

## Implementation steps

### Step 1: CitibikeAdapter

Follow the TfL adapter pattern: Polars for parse/aggregate, DuckDB only for
densification cross-join.

```python
CITIBIKE_BASE_URL = "https://s3.amazonaws.com/tripdata"

CITIBIKE_FILES: dict[str, str] = {
    "2023-12": "202312-citibike-tripdata.csv.zip",
    "2024-01": "202401-citibike-tripdata.csv.zip",
    "2024-02": "202402-citibike-tripdata.csv.zip",
}


class CitibikeAdapter(DatasetAdapter):
    name = "citibike"

    def download(self, raw_dir: Path, months: list[str]) -> list[Path]:
        # For each month:
        #   1. Download zip (atomic: .tmp -> rename) -- skip if zip exists
        #   2. Extract all CSVs from zip into raw_dir -- skip if already extracted
        #   3. Return list of extracted CSV paths
        # Idempotency: check extracted CSVs exist (not the zip),
        # since re-extraction is cheap but re-download is slow.

    def _parse_csv(self, csv_path: Path) -> pl.DataFrame:
        # Read CSV with Polars (try_parse_dates=False)
        # Rename: start_station_id -> zone_id, start_station_name -> zone_name
        # Filter: drop rows where zone_id is null or non-numeric
        #   - Use str.contains(r"^\d+$") to identify numeric IDs
        #   - Log count of dropped rows and why
        # Cast zone_id to Int64
        # Parse started_at: ISO format, truncate to hour
        #   - started_at is "2024-01-15 08:23:45" -- no UK date quirks
        # Return df.select(["zone_id", "zone_name", "hour_ts"])

    def _aggregate_hourly(self, trips: pl.DataFrame) -> pl.DataFrame:
        # group_by(["zone_id", "zone_name", "hour_ts"])
        # .agg(pl.len().alias("trip_count"))
        # .sort(["zone_id", "hour_ts"])
        # Identical to TfL adapter

    def prepare_hourly_history(
        self,
        raw_dir: Path,
        processed_dir: Path,
        config: dict,
    ) -> tuple[pl.DataFrame, list[int]]:
        months = config["data"]["months"]

        # 1. Collect all extracted CSVs for requested months
        #    For each month, list CSVs that came from that month's zip
        #    Raise FileNotFoundError if none found (same guard as TfL)

        # 2. Parse and concatenate all CSVs via _parse_csv()

        # 3. Log total trips loaded + trips dropped (non-numeric station IDs)

        # 4. Aggregate to hourly via _aggregate_hourly()

        # 5. Determine station universe: sorted unique zone_ids

        # 6. Build zone_name lookup from first occurrence

        # 7. Densify with DuckDB cross-join (zones x hours)
        #    Time range: 2023-12-01 00:00:00 to 2024-02-29 23:00:00
        #    LEFT JOIN hourly_agg, COALESCE trip_count to 0

        # 8. Join zone names, add null avg_fare + avg_distance columns

        # 9. Reorder to match HourlyHistorySchema:
        #    [zone_id, zone_name, hour_ts, trip_count, avg_fare, avg_distance]

        # 10. Return (dense_df, zone_ids)
```

Key detail for `download()` -- zip extraction:

```python
import zipfile

zip_path = raw_dir / zip_filename
with zipfile.ZipFile(zip_path) as zf:
    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
    for name in csv_names:
        dest = raw_dir / name
        if not dest.exists():
            zf.extract(name, raw_dir)
```

Multiple CSVs per zip are extracted side-by-side in `raw_dir`. The
`prepare_hourly_history` step globs for `*.csv` files matching the month
prefix (e.g. `202401*.csv`).

### Step 2: Config

```yaml
# configs/citibike.yaml
dataset:
  adapter: citibike

data:
  raw_dir: data/citibike/raw
  processed_dir: data/citibike/processed
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
    objective: regression
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
  experiment_name: "demandops-citibike"

artifacts:
  models_dir: artifacts/citibike/models
  reports_dir: artifacts/citibike/reports
  feature_schema_path: artifacts/citibike/models/feature_schema.json
  zone_universe_path: artifacts/citibike/zone_universe.json

serving:
  host: "0.0.0.0"
  port: 8002
  model_name: lightgbm
  history_path: data/citibike/processed/hourly_history.parquet
  feature_path: data/citibike/processed/features.parquet
  feature_schema_path: artifacts/citibike/models/feature_schema.json
  zone_universe_path: artifacts/citibike/zone_universe.json
  request_timeout_seconds: 10
```

Port 8002 avoids conflict with taxi (8000) and TfL (8001).

### Step 3: Tests

Mock CSV tests -- no network in CI.

**`test_citibike_parse`** -- Mock CSV with new-schema columns, verify correct
DataFrame output (zone_id, zone_name, hour_ts).

**`test_citibike_filter_non_numeric`** -- Rows with string station IDs
(`"AB123"`, `"6215.01"`) are dropped. Numeric IDs (`"5678"`) survive. Log
output includes drop count.

**`test_citibike_hourly_aggregation`** -- 5 trips at same station+hour produce
`trip_count=5`.

**`test_citibike_prepare_e2e`** -- Multi-file mock (simulating multi-CSV zip
extraction). Produces dense grid with correct shape: `n_stations * 2184`
hours. Verifies zero-fill for missing hours. Verifies `avg_fare` and
`avg_distance` are null. Validates against `HourlyHistorySchema`.

**`test_citibike_multi_csv_zip`** -- Zip containing 2 CSVs extracts and
processes both correctly.

**`test_missing_csv_raises_error`** -- `FileNotFoundError` if no CSVs found
for requested months.

### Step 4: Run pipeline + benchmark

```bash
python scripts/download_data.py --config configs/citibike.yaml
python scripts/prepare_data.py --config configs/citibike.yaml
python scripts/train.py --config configs/citibike.yaml
python scripts/benchmark.py --config configs/citibike.yaml
```

Produces `docs/benchmark_report_citibike.md`.

### Step 5: README + DECISIONS.md

Triple-dataset benchmark table in README:

| Metric             | NYC Taxi | London Bike-Share | NYC Bike-Share |
|--------------------|----------|-------------------|----------------|
| Zones/Stations     | 261      | 802               | ~1,500         |
| Grid rows          | 375K     | 1.75M             | ~3.3M          |
| Slot Mean MAE      | 3.40     | 0.75              | TBD            |
| LightGBM MAE       | 2.90     | 0.77              | TBD            |
| LightGBM vs Slot   | -14.6%   | +1.9%             | TBD            |

DECISIONS.md entry #21: Citibike adapter, string station ID filtering
rationale (Option C), cross-city cycling comparison result.

## Memory check

~1,500 stations x 2,184 hours x 8 columns x 8 bytes = ~210MB dense grid.
Fine on 16GB RAM.

## Risk

**Low.** Same pattern as TfL. Citibike CSV schema is cleaner (ISO dates, no
format inconsistency). New wrinkles: non-numeric station IDs (handled by
filtering) and multi-CSV zips (handled by extraction loop). Both are tested.
