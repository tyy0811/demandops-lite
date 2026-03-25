# demandops-lite

![CI](https://github.com/tyy0811/demandops-lite/actions/workflows/ci.yaml/badge.svg)

LightGBM beats two honest baselines by 14.6–27.7% MAE on NYC taxi demand — served via FastAPI with train-serve feature parity, Pandera contracts, and Prometheus monitoring. 261 zones, 375K rows, 99 tests.

End-to-end demand prediction pipeline for NYC taxi data — from data contracts through honest baselines to lag-aware one-step-ahead monitored inference.

> **99 tests | 261 zones | 375K rows | Prometheus `/metrics` | Docker ready**

## Benchmark Results

**261 zones | 375,840 rows | 9 features | Test: Feb 15–29, 2024**

| Model | MAE | RMSE | vs Slot Mean | vs Seasonal Naive |
|-------|-----|------|-------------|-------------------|
| **LightGBM** | **2.90** | **9.37** | **-14.6%** | **-27.7%** |
| Slot Mean | 3.40 | 12.12 | — | — |
| Seasonal Naive | 4.01 | 13.99 | +18.1% | — |

LightGBM reduces MAE by 14.6% vs slot mean and 27.7% vs seasonal naive. 1.4% of predictions clipped to zero. Lag features (1h, 168h, 24h) dominate feature importance. Hardest zones: JFK Airport (MAE 29.9), Midtown Center (MAE 25.0).

> **sMAPE note:** LightGBM's sMAPE (138.6%) exceeds both baselines (108.8%, 99.3%) because sMAPE heavily penalizes small absolute errors on near-zero actuals — a known artifact on zero-heavy distributions. MAE and RMSE are the appropriate metrics for this task. Full sMAPE breakdown in the benchmark report.

Full report: [`docs/benchmark_report.md`](docs/benchmark_report.md)

## V1 → V2 Improvements

| Feature | V1 | V2 | Signal |
|---------|----|----|--------|
| CI quality gate | Lint + tests | + MAE regression gate + Docker smoke test | ML-specific CI |
| Batch inference | Single-record only | `/predict/batch` (up to 10K) | Production serving |
| Objective selection | regression (implicit) | regression vs. Poisson (documented) | Scientific rigor |
| API contract | model_name only | + model_version, model_objective | Serving observability |

See [DECISIONS.md](DECISIONS.md) for the reasoning behind each design choice.

## Dual-Dataset Benchmark

Same pipeline, two datasets, two cities:

| Metric | NYC Taxi | London Bike-Share |
|--------|----------|-------------------|
| Zones/Stations | 261 | 802 |
| Grid rows | 375K | 1.75M |
| Feature rows | 289K | 1.15M |
| Slot Mean MAE | 3.40 | **0.75** |
| LightGBM MAE | **2.90** | 0.77 |
| LightGBM vs Slot Mean | -14.6% | +1.9% |

LightGBM dominates on NYC taxi data (high-variance demand, 14.6% MAE reduction). On London bike-share, the simpler slot mean is competitive — low-variance station demand means the historical average is hard to beat on MAE, though LightGBM wins on RMSE (1.28 vs 1.31). Both datasets use identical feature engineering, model training, and evaluation code via the DatasetAdapter pattern.

## Quick Start

```bash
# 1. Create virtual environment and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

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
NYC TLC Parquet → DuckDB (filter/aggregate/densify) → Polars (features) → LightGBM (predict)
                                                                              ↓
                                                                         FastAPI /predict
                                                                              ↑
                                                             FeatureService (lag reconstruction)
```

**Data flow:**
1. **Download** 3 months of NYC Yellow Taxi trip data (Dec 2023 – Feb 2024)
2. **Prepare** dense zone×hour grid via DuckDB, engineer lag/temporal features with Polars
3. **Validate** at every boundary with Pandera data contracts
4. **Train** slot mean baseline, seasonal naive baseline, and LightGBM
5. **Evaluate** on held-out test set with per-zone and edge-case analysis
6. **Serve** via FastAPI with FeatureService reconstructing lags at request time

## What This Demonstrates

- **Data engineering**: DuckDB SQL aggregation, dense grid construction, Polars feature pipelines
- **Data contracts**: Pandera validation at every pipeline boundary
- **ML lifecycle**: Temporal split (half-open), two honest baselines, MLflow tracking, objective experiments
- **Train-serve parity**: FeatureService reconstructs identical lag features at inference time
- **Batch inference**: `/predict/batch` for up to 10K vectorized predictions per request
- **Monitoring**: Prometheus counters, latency histograms, prediction distribution
- **Production patterns**: Docker, CI, structured logging, config-driven, graceful degradation

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
├── configs/default.yaml          # All configuration (splits, models, paths)
├── demandops/
│   ├── features.py               # FEATURE_COLUMNS — single source of truth
│   ├── data/
│   │   ├── download.py           # Atomic TLC data download
│   │   ├── prepare.py            # DuckDB → Polars pipeline
│   │   ├── schemas.py            # Pandera data contracts
│   │   └── splits.py             # Half-open temporal split
│   ├── models/
│   │   ├── registry.py           # DemandModel ABC + factory
│   │   ├── baselines.py          # Slot mean, seasonal naive
│   │   └── lightgbm_model.py     # LightGBM with clipping + predict_raw
│   ├── training/
│   │   ├── train.py              # Train all models, MLflow logging
│   │   └── evaluate.py           # Metrics, per-zone analysis, edge cases
│   ├── serving/
│   │   ├── app.py                # FastAPI factory (graceful degradation)
│   │   ├── routes.py             # /predict, /predict/batch, /health, /metrics
│   │   ├── feature_service.py    # Lag reconstruction from dense history
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── metrics.py            # Prometheus counters/histograms
│   │   └── middleware.py         # Request logging
│   └── monitoring/
│       └── checks.py             # Sparse zone, extreme prediction checks
├── scripts/                      # CLI entrypoints
├── tests/                        # 99 tests
├── docker/                       # Dockerfile + docker-compose
├── .github/workflows/ci.yaml     # GitHub Actions
└── Makefile                      # Pipeline targets
```

## Data Pipeline

**Input:** NYC TLC Yellow Taxi trip records (Dec 2023 – Feb 2024)

**Grid:** Dense zone×hour matrix. Every (zone_id, hour_ts) pair has exactly one row. Zero-demand hours are filled with trip_count=0. December provides warm-up for lag features.

**Features (9):**

| Feature | Source | Description |
|---------|--------|-------------|
| hour_of_day | temporal | 0–23 |
| day_of_week | temporal | 0=Mon, 6=Sun (Python convention) |
| is_weekend | temporal | Sat(5) + Sun(6) |
| month | temporal | 1–12 |
| zone_id | categorical | TLC pickup location ID |
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

Run `make benchmark` after `make pipeline` to generate `docs/benchmark_report.md` with full model comparison, feature importance, edge-case analysis, and per-zone error breakdown.

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

## Development

```bash
make test       # Run 99 tests
make lint       # ruff check + format --check
make format     # Auto-fix formatting
make clean      # Remove generated data/artifacts
```

## V2 Roadmap

- [x] Batch prediction endpoint (`/predict/batch`, up to 10K records)
- [x] Poisson objective experiment (regression wins by 0.2% MAE; documented)
- [x] MAE regression gate in CI (frozen fixture + threshold assertion)
- [x] Second dataset (TfL Cycle Hire) for pipeline generality

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for the full rationale behind:
- DuckDB for aggregation (not pandas)
- Polars for feature engineering (not pandas)
- Dense grid with December warm-up (not sparse)
- Weekday convention: 0=Mon, 6=Sun everywhere
- LightGBM predictions clipped to zero (Poisson tested, regression wins by 0.2% MAE)
- joblib for model serialization (not Booster text format)
- `app.state` for dependency injection (not module globals)
- Pandera validation at every pipeline boundary
