# demandops-lite

End-to-end demand prediction pipeline for NYC taxi data — from data contracts through honest baselines to lag-aware one-step-ahead monitored inference.

Demonstrates ML engineering best practices on CPU: DuckDB for aggregation, Polars for feature engineering, LightGBM for prediction, FastAPI for serving, and Pandera for validation at every pipeline boundary.

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
│   │   ├── routes.py             # /predict, /health, /metrics
│   │   ├── feature_service.py    # Lag reconstruction from dense history
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── metrics.py            # Prometheus counters/histograms
│   │   └── middleware.py         # Request logging
│   └── monitoring/
│       └── checks.py             # Sparse zone, extreme prediction checks
├── scripts/                      # CLI entrypoints
├── tests/                        # 77 tests
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

```json
{"zone_id": 161, "hour_ts": "2024-02-20T14:00:00"}
```

Returns predicted trip count with metadata (latency, features used, request ID).

**Rejection (422):** unsupported zone_id or hour_ts outside [2024-01-01, 2024-03-01).

### GET /health

Returns model/history load status. Reports `"degraded"` if artifacts are missing — the app starts but cannot serve predictions.

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
make test       # Run 77 tests
make lint       # ruff check + format --check
make format     # Auto-fix formatting
make clean      # Remove generated data/artifacts
```

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for the full rationale behind:
- DuckDB for aggregation (not pandas)
- Polars for feature engineering (not pandas)
- Dense grid with December warm-up (not sparse)
- Weekday convention: 0=Mon, 6=Sun everywhere
- LightGBM predictions clipped to zero (not constrained objective)
- joblib for model serialization (not Booster text format)
- `app.state` for dependency injection (not module globals)
- Pandera validation at every pipeline boundary
