# Design Decisions

Rationale behind the key technical choices in demandops-lite (V1 and V2).

## 1. DuckDB for Aggregation

**Decision:** Use DuckDB (in-process) for raw parquet loading, filtering, hourly aggregation, and grid densification.

**Why:** DuckDB handles multi-file parquet reads, SQL-based filtering, and cross-join grid densification natively. No need to load raw trip data into Python memory. The entire aggregation runs as SQL, keeping the prepare script focused on orchestration rather than pandas gymnastics.

**Alternative considered:** pandas. Rejected because pandas requires loading all raw data into memory, and the aggregation/densification logic is more naturally expressed in SQL.

## 2. Polars for Feature Engineering

**Decision:** Use Polars (not pandas) for lag computation, temporal features, and rolling means.

**Why:** Polars has explicit sort-then-shift semantics, lazy evaluation, and clear `.over()` partitioning. The `group_by().map_groups()` pattern gives unambiguous per-group rolling computation without version-dependent behavior. Polars also avoids the pandas copy-on-write ambiguity.

**Alternative considered:** pandas with groupby().transform(). Rejected because rolling window behavior over grouped data in pandas has historically been a source of subtle bugs.

## 3. Dense Grid with December Warm-up

**Decision:** Build a dense zone×hour grid covering Dec 2023 – Feb 2024. Fill zero-demand hours with trip_count=0. December is warm-up only (dropped before training).

**Why:** A dense grid guarantees that all lag features (1h, 24h, 168h) and rolling means are computable without null handling. Without densification, sparse zones would have missing lags that require imputation — a source of train-serve skew. December provides the 168h (one week) warm-up so January rows have complete lag_168h values.

## 4. Weekday Convention: 0=Mon, 6=Sun

**Decision:** Use Python's `datetime.weekday()` convention (0=Mon, 6=Sun) everywhere. Polars `dt.weekday()` returns 1–7; we subtract 1 immediately.

**Why:** A single convention eliminates a class of off-by-one bugs. Python's convention is the natural choice because `FeatureService` uses `datetime.weekday()` at serving time. Making the training pipeline match (by subtracting 1 from Polars) ensures train-serve parity. `is_weekend = day_of_week >= 5` is correct under both.

## 5. LightGBM Predictions Clipped to Zero (Validated by Experiment)

**Decision:** `predict()` returns `np.clip(raw, 0.0, None)`. `predict_raw()` returns unclipped values for tracking clip statistics. Default objective remains `regression` (L2).

**Why:** Trip counts are non-negative by definition. LightGBM's regression objective can produce small negative predictions near zero. Clipping is simpler and more interpretable than a constrained objective (Poisson, Tweedie). The `predict_raw()` method allows accurate tracking of how many predictions required clipping — important for monitoring model quality.

**Empirical validation:** Ran a controlled experiment comparing regression vs Poisson objectives with identical hyperparameters on the same train/val/test split (`scripts/compare_objectives.py`):

| Objective | Test MAE | Negative Raw Preds | Best Iteration | Time |
|-----------|----------|-------------------|----------------|------|
| regression | 2.8997 | 1,344 (1.4%) | 455 | 2.8s |
| poisson | 2.9066 | 0 (0.0%) | 500 | 4.0s |

Regression wins by 0.2% MAE. Poisson eliminates all negative predictions, but 1.4% clip rate is negligible in practice and handled transparently by `np.clip`. Both runs logged to MLflow experiment `objective-comparison`.

**Alternative considered:** Poisson regression objective. Empirically tested and rejected: 0.2% worse MAE, 43% slower training, and the problem Poisson solves (negative predictions) affects only 1.4% of outputs and is already handled by clipping.

## 6. joblib for Model Serialization

**Decision:** Save/load LightGBM models via `joblib.dump()`/`joblib.load()` on the full `LGBMRegressor` object.

**Why:** The sklearn wrapper's internal `_Booster` attribute is not a reliable serialization boundary. `joblib` preserves the full fitted state including `best_iteration_`, `feature_importances_`, and all hyperparameters. Round-trip fidelity is verified by test.

**Alternative considered:** `model.booster_.save_model()` (Booster text format). Rejected because it loses the sklearn wrapper state and requires manual reconstruction.

## 7. app.state for Dependency Injection

**Decision:** Store `FeatureService`, model, and metadata on `app.state` via a `configure()` function. No module-level globals.

**Why:** Module-level globals create import-time side effects and make testing difficult. `app.state` is FastAPI's recommended pattern — each test can create a fresh app with different dependencies (mock service, mock model, degraded state) without monkeypatching.

## 8. Pandera Validation at Every Boundary

**Decision:** Validate DataFrames against Pandera schemas before writing parquet files. Use `DataFrameModel` with `Config(coerce=True)`.

**Why:** Declared-but-not-enforced schemas are worse than no schemas — they give false confidence. Validation before write means corrupted data never reaches disk. `coerce=True` handles DuckDB's Int32 and Polars' Int8 outputs without requiring explicit casts in the pipeline.

## 9. Half-Open Temporal Splits

**Decision:** All split boundaries use `>= start` and `< end`. `train_end == val_start`, `val_end == test_start`.

**Why:** Half-open intervals guarantee no gaps and no overlaps by construction. The boundary timestamp belongs to exactly one split. This is verifiable by asserting that `len(train) + len(val) + len(test) == len(df)`.

## 10. Feature Schema Artifact

**Decision:** Save `feature_schema.json` during training. `FeatureService` validates its column order against `FEATURE_COLUMNS` at startup.

**Why:** The persisted schema is the contract between training and serving. If code and artifact diverge (e.g., a feature is added to `FEATURE_COLUMNS` but the model was trained on the old set), the startup validation catches it immediately instead of silently serving predictions with wrong feature alignment.

## 11. Slot Mean Baseline Keys by Zone

**Decision:** `HistoricalSlotMean` keys by `(zone_id, hour_of_day, day_of_week)`, not just `(hour_of_day, day_of_week)`.

**Why:** The prediction target is demand per zone per hour. A baseline that ignores zone_id collapses all zones to a single mean per time slot — useless for a per-zone prediction task. Different zones have fundamentally different demand patterns (JFK vs. residential Brooklyn).

## 12. Graceful Degradation

**Decision:** The serving app starts even when artifacts (history, model, schema) are missing. `/health` reports `"degraded"` instead of crashing.

**Why:** In production, the app process should be observable even before artifacts are deployed. A hard crash on missing files prevents health checks, readiness probes, and diagnostic endpoints from functioning. Degraded state is preferable to no state.

## 13. Atomic Downloads

**Decision:** Downloads write to a `.tmp` file and rename on success. Interrupted downloads are cleaned up.

**Why:** `urlretrieve()` writes directly to the destination path. An interrupted download leaves a partial file that `dest.exists()` treats as complete on the next run. Atomic rename ensures only complete files are visible to the pipeline.

## 14. Timezone Normalization

**Decision:** `FeatureService.get_features()` converts timezone-aware timestamps to UTC before stripping `tzinfo` for lookup.

**Why:** Pydantic may parse `"2024-02-01T12:00:00+02:00"` as timezone-aware. Simply stripping `tzinfo` (`.replace(tzinfo=None)`) would treat it as 12:00 UTC instead of 10:00 UTC, producing wrong lag features. `astimezone(UTC)` first, then strip, gives correct behavior.

## 15. Frozen Zone Universe

**Decision:** The zone universe is determined once during `prepare()` from the distinct pickup location IDs in the raw data, saved to `zone_universe.json`, and treated as immutable at serving time. Requests for zone IDs not in the universe are rejected with 422.

**Why:** The dense grid and all lag features are built over a fixed set of zones. A zone that wasn't in the training data has no history, no lag values, and no slot mean — any prediction for it would be meaningless. Rejecting unsupported zones with an explicit error (including the supported zone count) is more honest than returning a fallback prediction with no statistical backing. The zone universe file is the shared contract between prepare, train, and serve.

**Alternative considered:** Accepting any zone_id in 1–263 and returning a global-mean fallback. Rejected because it hides a data gap behind a plausible-looking number.

## 16. prometheus-client for Monitoring

**Decision:** Use the official `prometheus-client` library with module-level metric definitions (Counters, Histograms, Gauges) and a `/metrics` endpoint returning `generate_latest()`.

**Why:** Prometheus is the de facto standard for ML serving observability. Module-level metric definitions are the library's intended pattern — they register once on the default `CollectorRegistry` at import time. The `/metrics` endpoint returns the standard text exposition format, compatible with any Prometheus scraper without additional infrastructure. Metrics cover: request counts by endpoint/status, prediction latency, prediction value distribution, rejection reasons, error counts, and model/history load status gauges.

**Alternative considered:** OpenTelemetry. Rejected for V1 because it adds complexity (exporters, collectors, SDK configuration) without a clear benefit when the deployment target is a single-node Docker container with a Prometheus scrape.

## 17. MAE Regression Gate in CI

**Decision:** Add a frozen test set (7,200 rows, 20 zones, committed to repo) and an assertion that MAE stays below 3.20 on every push. The pre-trained model artifact (~1.1MB) is also committed.

**Why:** ML pipelines have a subtle failure mode: code changes that pass all unit tests but silently degrade model quality. A renamed column, a changed default, or a dependency upgrade can shift predictions without any test catching it. The regression gate catches this class of bug.

The threshold (3.20) is 10% above the V1 baseline MAE of 2.90 — deliberately loose to catch regressions, not block improvements. The frozen test set never changes: if the evaluation anchor evolves with the code, it can't detect drift. The training set and features may change; the evaluation anchor must not.

The test loads the raw `LGBMRegressor` (not the `LightGBMModel` wrapper) intentionally — testing raw predictions catches regressions before the serving layer's `np.clip(0)` masks them. A non-negative prediction check acts as an early warning for when to reconsider the Poisson objective.

**Alternative considered:** `pytest.skip` when model not found (gate only runs locally). Rejected because a gate that depends on the developer remembering to run it is exactly the failure mode CI exists to prevent.

## 18. Batch Prediction Endpoint

**Decision:** `POST /predict/batch` accepts up to 10,000 records. Loops over `FeatureService.get_features()` (one dict lookup per lag per request) then calls `model.predict()` once on the full feature matrix.

**Why:** Demand forecasting is a batch operation: 261 zones × 24 hours = 6,264 predictions. Exposing this as a single endpoint avoids 6,264 HTTP round-trips.

All-or-nothing validation: if any request has an unsupported zone or timestamp, the entire batch returns 422 with the first error. Partial success was considered but rejected — it requires per-item status codes, a different response schema, and decisions about whether to run inference on the valid subset. None of this adds meaningful signal for the portfolio use case.

`/predict` stays independent from `/predict/batch`. The original plan suggested wrapping single through batch, but each endpoint has its own request ID, error handling, and Prometheus labeling. Wrapping adds indirection for zero code savings.

Performance: 10K requests × 27 dict lookups = 270K O(1) lookups, well under 100ms. Vectorized `model.predict()` on a 10K feature matrix is ~5ms on CPU. Full-request `latency_ms` is reported in the response (timer starts before the feature loop).
