## Benchmark Results — NYC Taxi Demand Prediction

**Target:** Hourly trip count per pickup zone
**Entities:** 261 (from zone_universe.json)
**Grid:** 570,024 rows (261 pickup zones × hourly)
**Train:** [2024-01-01, 2024-02-01) | **Val:** [2024-02-01, 2024-02-15) | **Test:** [2024-02-15, 2024-03-01)
**Features:** 9 (temporal + lag)

**Negative prediction handling:** LightGBM predictions clipped to zero (1344 predictions, 1.4%)

### Model Comparison

| Model | MAE | RMSE | sMAPE | Latency (ms) |
|-------|-----|------|-------|-------------|
| slot_mean | 3.40 | 12.12 | 108.80% | 116.0 |
| seasonal_naive | 4.01 | 13.99 | 99.27% | 0.1 |
| lightgbm | 2.90 | 9.37 | 138.61% | 601.5 |
| **vs Slot Mean** | -14.6% | — | — | — |
| **vs Seasonal Naive** | -27.7% | — | — | — |

### Feature Importance (LightGBM, top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | lag_1h | 2093.0000 |
| 2 | lag_168h | 2050.0000 |
| 3 | lag_24h | 2026.0000 |
| 4 | hour_of_day | 1845.0000 |
| 5 | rolling_mean_24h | 1526.0000 |
| 6 | zone_id | 1193.0000 |
| 7 | day_of_week | 1130.0000 |
| 8 | is_weekend | 66.0000 |
| 9 | month | 0.0000 |

### Edge-Case Analysis

| Segment | Definition | N rows | Slot Mean MAE | LightGBM MAE | Δ (LightGBM vs Slot Mean) |
|---------|------------|--------|---------------|--------------|---------------------------|
| sparse_zones | Zones with mean demand < P10 | 9360 | 0.01 | 0.07 | +1038.7% |
| dense_zones | Zones with mean demand > P90 | 9360 | 21.06 | 17.38 | -17.5% |
| late_night | Hours 0–5 | 23490 | 2.07 | 1.49 | -27.7% |
| peak_hours | Hours 7–9, 17–19 | 23490 | 4.14 | 3.70 | -10.6% |
| weekend | Saturday + Sunday (day_of_week >= 5) | 25056 | 3.53 | 2.95 | -16.6% |
| weekday | Monday–Friday (day_of_week < 5) | 68904 | 3.35 | 2.88 | -13.8% |
| zero_demand | Hours with trip_count == 0 | 54962 | 0.21 | 0.28 | +31.3% |

### Hardest Pickup Zones (by LightGBM MAE)

| ID | Name | MAE | Mean Demand |
|----|------|-----|-------------|
| 132 | JFK Airport | 29.92 | 189.30 |
| 161 | Midtown Center | 25.00 | 213.49 |
| 186 | Penn Station/Madison Sq West | 24.34 | 146.38 |
| 138 | LaGuardia Airport | 24.22 | 132.35 |
| 236 | Upper East Side North | 23.58 | 186.28 |

### Configuration

```yaml
train_start: 2024-01-01T00:00:00
train_end: 2024-02-01T00:00:00
val_end: 2024-02-15T00:00:00
test_end: 2024-03-01T00:00:00
n_zones: 261
lightgbm.objective: regression
lightgbm.n_estimators: 500
lightgbm.learning_rate: 0.05
lightgbm.max_depth: 6
lightgbm.num_leaves: 31
lightgbm.min_child_samples: 20
lightgbm.subsample: 0.8
lightgbm.colsample_bytree: 0.8
lightgbm.early_stopping_rounds: 50
lightgbm.random_state: 42
lightgbm.num_threads: -1
```

### MLflow Run IDs

| Model | Run ID |
|-------|--------|
| slot_mean | `31cc9014bfc643729a3d5d72972de96d` |
| seasonal_naive | `e5b91e7e67e349c9ad87ceb370b0d1ac` |
| lightgbm | `eaeff1be5ec44b79a7c40afe0a7af19a` |
