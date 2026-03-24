# Design Decisions

Rationale behind the key technical choices in demandops-lite V1.

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

## 5. LightGBM Predictions Clipped to Zero

**Decision:** `predict()` returns `np.clip(raw, 0.0, None)`. `predict_raw()` returns unclipped values for tracking clip statistics.

**Why:** Trip counts are non-negative by definition. LightGBM's regression objective can produce small negative predictions near zero. Clipping is simpler and more interpretable than a constrained objective (Poisson, Tweedie). The `predict_raw()` method allows accurate tracking of how many predictions required clipping — important for monitoring model quality.

**Alternative considered:** Poisson regression objective. Rejected because it constrains the entire prediction surface for what is a boundary-only problem. Clipping is transparent and reversible.

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
