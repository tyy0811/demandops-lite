# demandops-lite

![CI](https://github.com/tyy0811/demandops-lite/actions/workflows/ci.yaml/badge.svg)

Same pipeline, three datasets, two cities: NYC taxi (261 zones) + London bike-share (802 stations) + NYC bike-share (2,144 stations). LightGBM, FastAPI, Pandera contracts, Prometheus monitoring, 112 tests.

End-to-end demand prediction pipeline with DatasetAdapter pattern — from data contracts through honest baselines to lag-aware one-step-ahead monitored inference.

> **112 tests | 3 datasets | 3,207 zones/stations | Prometheus `/metrics` | Docker ready**

## Triple-Dataset Benchmark

Same pipeline, three datasets, two cities:

| Metric | NYC Taxi | London Bike-Share | NYC Bike-Share |
|--------|----------|-------------------|----------------|
| Zones/Stations | 261 | 802 | 2,144 |
| Grid rows | 375K | 1.75M | 4.68M |
| Feature rows | 289K | 1.15M | 3.09M |
| Slot Mean MAE | 3.40 | **0.75** | 1.03 |
| LightGBM MAE | **2.90** | 0.77 | **0.95** |
| LightGBM vs Slot Mean | -14.6% | +1.9% | -7.7% |

LightGBM dominates on NYC taxi data (high-variance demand, 14.6% MAE reduction) and NYC bike-share (7.7% improvement over slot mean across 2,144 stations). On London bike-share, the simpler slot mean is competitive — low-variance station demand means the historical average is hard to beat on MAE, though LightGBM wins on RMSE (1.28 vs 1.31). The Citibike result lands between the two: more stations and more variable demand than TfL, but less variance than taxi — LightGBM's advantage scales with demand heterogeneity. All three datasets use identical feature engineering, model training, and evaluation code via the DatasetAdapter pattern.

Full reports: [`docs/benchmark_report.md`](docs/benchmark_report.md) | [`docs/benchmark_report_tfl.md`](docs/benchmark_report_tfl.md) | [`docs/benchmark_report_citibike.md`](docs/benchmark_report_citibike.md)

## What This Demonstrates

- **Pipeline generality**: DatasetAdapter pattern — same code runs on NYC taxi, London bike-share, and NYC bike-share
- **Data engineering**: DuckDB SQL aggregation, dense grid construction, Polars feature pipelines
- **Data contracts**: Pandera validation at every pipeline boundary
- **ML lifecycle**: Temporal split (half-open), two honest baselines, MLflow tracking, objective experiments
- **Train-serve parity**: FeatureService reconstructs identical lag features at inference time
- **Batch inference**: `/predict/batch` for up to 10K vectorized predictions per request
- **ML quality gates**: MAE regression gate in CI with frozen test fixture
- **Monitoring**: Prometheus counters, latency histograms, prediction distribution
- **Production patterns**: Docker, CI, structured logging, config-driven, graceful degradation

## V1 → V2 Improvements

| Feature | V1 | V2 | Signal |
|---------|----|----|--------|
| Datasets | NYC taxi only | + London bike-share + NYC bike-share (DatasetAdapter) | Pipeline generality |
| CI quality gate | Lint + tests | + MAE regression gate + Docker smoke test | ML-specific CI |
| Batch inference | Single-record only | `/predict/batch` (up to 10K) | Production serving |
| Objective selection | regression (implicit) | regression vs. Poisson (documented) | Scientific rigor |
| API contract | model_name only | + model_version, model_objective | Serving observability |

See [DECISIONS.md](DECISIONS.md) for the reasoning behind each design choice.

## Quick Start

```bash
# 1. Create virtual environment and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1b. (Optional) Install dbt analytics layer
pip install -e ".[dbt]"

# 2. Run full pipeline (download → prepare → train → evaluate → benchmark)
make pipeline

# 3. Start the API server
make serve

# 4. Make a prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"zone_id": 161, "hour_ts": "2024-02-20T14:00:00"}'

# 5. Check health
curl http://localhost:8001/health

# 6. View Prometheus metrics
curl http://localhost:8001/metrics
```

## Architecture

```
Raw data → DatasetAdapter (download/aggregate/densify) → Polars (features) → LightGBM (predict)
                                                                                  ↓
                                                                             FastAPI /predict
                                                                                  ↑
                                                                 FeatureService (lag reconstruction)
```

**Data flow:**
1. **Download** raw trip data via DatasetAdapter (NYC taxi parquets, TfL bike-share CSVs, or Citibike zipped CSVs)
2. **Prepare** dense entity×hour grid via DuckDB, engineer lag/temporal features with Polars
3. **Validate** at every boundary with Pandera data contracts
4. **Train** slot mean baseline, seasonal naive baseline, and LightGBM
5. **Evaluate** on held-out test set with per-zone and edge-case analysis
6. **Serve** via FastAPI with FeatureService reconstructing lags at request time

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Aggregation | DuckDB |
| Feature Engineering | Polars |
| Validation | Pandera (DataFrameModel) |
| Prediction | LightGBM (clipped to zero) |
| Experiment Tracking | MLflow |
| Serving | FastAPI + Pydantic v2 |
| Monitoring | prometheus-client |
| Serialization | joblib |
| Logging | structlog |
| Testing | pytest |
| Linting | ruff |
| CI | GitHub Actions |
| Container | Docker |

## Project Structure

```
demandops-lite/
├── configs/
│   ├── default.yaml             # NYC taxi configuration
│   ├── tfl.yaml                 # London bike-share configuration
│   └── citibike.yaml            # NYC bike-share configuration
├── demandops/
│   ├── features.py              # FEATURE_COLUMNS — single source of truth
│   ├── data/
│   │   ├── adapters/
│   │   │   ├── base.py          # DatasetAdapter ABC
│   │   │   ├── taxi.py          # NYC TLC Yellow Taxi adapter
│   │   │   ├── tfl.py           # TfL Santander Cycle Hire adapter
│   │   │   └── citibike.py      # Citibike NYC adapter
│   │   ├── download.py          # Adapter-delegated download
│   │   ├── prepare.py           # Shared feature engineering pipeline
│   │   ├── schemas.py           # Pandera data contracts
│   │   └── splits.py            # Half-open temporal split
│   ├── models/
│   │   ├── registry.py          # DemandModel ABC + factory
│   │   ├── baselines.py         # Slot mean, seasonal naive
│   │   └── lightgbm_model.py    # LightGBM with clipping + predict_raw
│   ├── training/
│   │   ├── train.py             # Train all models, MLflow logging
│   │   └── evaluate.py          # Metrics, per-zone analysis, edge cases
│   ├── serving/
│   │   ├── app.py               # FastAPI factory (graceful degradation)
│   │   ├── routes.py            # /predict, /predict/batch, /health, /metrics
│   │   ├── feature_service.py   # Lag reconstruction from dense history
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── metrics.py           # Prometheus counters/histograms
│   │   └── middleware.py        # Request logging
│   └── monitoring/
│       └── checks.py            # Sparse zone, extreme prediction checks
├── scripts/                     # CLI entrypoints (all accept --config)
├── tests/                       # 112 tests
├── docker/                      # Dockerfile + docker-compose
├── .github/workflows/ci.yaml   # GitHub Actions (test + Docker smoke test)
└── Makefile                     # Pipeline targets
```

## Data Pipeline

**Input:** Trip records via DatasetAdapter (NYC taxi parquets, TfL bike-share CSVs, or Citibike zipped CSVs)

**Grid:** Dense entity×hour matrix. Every (zone_id, hour_ts) pair has exactly one row. Zero-demand hours are filled with trip_count=0. December provides warm-up for lag features.

**Features (9):**

| Feature | Source | Description |
|---------|--------|-------------|
| hour_of_day | temporal | 0–23 |
| day_of_week | temporal | 0=Mon, 6=Sun (Python convention) |
| is_weekend | temporal | Sat(5) + Sun(6) |
| month | temporal | 1–12 |
| zone_id | categorical | Entity ID (taxi zone or bike station) |
| lag_1h | lag | trip_count at t-1 |
| lag_24h | lag | trip_count at t-24 |
| lag_168h | lag | trip_count at t-168 (same hour, 1 week ago) |
| rolling_mean_24h | rolling | mean of trip_count over [t-24, t-1] |

**Splits (half-open):**
- Train: [2024-01-01, 2024-02-01)
- Val: [2024-02-01, 2024-02-15)
- Test: [2024-02-15, 2024-03-01)

## Models

| Model | Description |
|-------|-------------|
| **Slot Mean** | Historical mean per (zone_id, hour_of_day, day_of_week) |
| **Seasonal Naive** | lag_168h (same hour, same day, one week ago) |
| **LightGBM** | Gradient-boosted trees, predictions clipped to zero |

Run `python scripts/benchmark.py --config configs/default.yaml` for NYC taxi, `--config configs/tfl.yaml` for London, or `--config configs/citibike.yaml` for NYC bike-share.

## API

### POST /predict

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"zone_id": 161, "hour_ts": "2024-02-20T14:00:00"}'
```

Response:

```json
{
  "zone_id": 161,
  "zone_name": "Midtown Center",
  "hour_ts": "2024-02-20T14:00:00",
  "predicted_count": 42.3,
  "model_name": "lightgbm",
  "model_version": "lightgbm-regression",
  "metadata": {
    "latency_ms": 12.4,
    "request_id": "abc-123",
    "features_used": {"lag_1h": 38.0, "lag_24h": 41.0, "lag_168h": 44.0},
    "input_warnings": []
  }
}
```

**Rejection (422):** unsupported zone_id or hour_ts outside [2024-01-01, 2024-03-01).

### POST /predict/batch

Up to 10,000 predictions in a single request. All-or-nothing validation: one bad zone_id fails the entire batch.

```bash
curl -X POST http://localhost:8001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [
    {"zone_id": 161, "hour_ts": "2024-02-20T14:00:00"},
    {"zone_id": 162, "hour_ts": "2024-02-20T14:00:00"},
    {"zone_id": 163, "hour_ts": "2024-02-20T14:00:00"}
  ]}'
```

Response:

```json
{
  "predictions": [
    {"zone_id": 161, "predicted_count": 42.3, "model_name": "lightgbm", "model_version": "lightgbm-regression", "...": "..."},
    {"zone_id": 162, "predicted_count": 18.1, "...": "..."},
    {"zone_id": 163, "predicted_count": 7.5, "...": "..."}
  ],
  "prediction_count": 3,
  "latency_ms": 15.2
}
```

### GET /health

Returns model/history load status, model objective, version, and supported zones. Reports `"degraded"` if artifacts are missing — the app starts but cannot serve predictions.

### GET /metrics

Prometheus-format metrics: request counts, latency histograms, prediction value distribution, rejection/error counters.

## Docker

```bash
# Build and run (requires make pipeline first for artifacts)
cd docker && docker-compose up --build -d

# Test
curl http://localhost:8001/health

# Stop
docker-compose down
```

## Analytics Engineering (dbt)

The SQL transformation logic (filter → aggregate → densify) is also expressed
as a dbt project using dbt-duckdb. This provides:

- **Versioned SQL models** with staging → intermediate → mart lineage
- **Schema tests** (not_null, unique composite keys, accepted_values, no hour gaps)
- **Auto-generated documentation** (`dbt docs generate`)

The dbt layer produces the same dense zone×hour grid as `prepare.py` —
verified by parity checks on row count, trip_count, avg_fare, avg_distance,
and temporal features. Lag and rolling features are computed downstream in Polars.

```bash
make dbt-install
make dbt-all        # run + test (28 tests)
make dbt-docs       # browse at http://localhost:8080
```

## Development

```bash
make test       # Run 112 tests
make lint       # ruff check + format --check
make format     # Auto-fix formatting
make clean      # Remove generated data/artifacts
```

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for 23 documented rationales, including:
- DatasetAdapter pattern for pipeline generality (NYC taxi + London bike-share + NYC bike-share)
- DuckDB for aggregation, Polars for feature engineering (not pandas)
- Poisson vs regression objective (empirically tested, regression wins by 0.2%)
- MAE regression gate in CI (frozen fixture + committed model)
- Batch endpoint design (all-or-nothing validation, independent from /predict)
- Dense grid with December warm-up, half-open temporal splits
- Graceful degradation, Pandera at every boundary, Prometheus monitoring
