## Benchmark Results — NYC Bike-Share Demand Prediction

**Target:** Hourly trip count per station
**Entities:** 2144 (from zone_universe.json)
**Grid:** 4,682,496 rows (2144 stations × hourly)
**Train:** [2024-02-01, 2024-03-01) | **Val:** [2024-03-01, 2024-03-16) | **Test:** [2024-03-16, 2024-04-01)
**Features:** 9 (temporal + lag)

**Negative prediction handling:** LightGBM predictions clipped to zero (21741 predictions, 2.6%)

### Model Comparison

| Model | MAE | RMSE | sMAPE | Latency (ms) |
|-------|-----|------|-------|-------------|
| slot_mean | 1.03 | 2.12 | 121.96% | 899.2 |
| seasonal_naive | 1.33 | 2.89 | 120.59% | 2.2 |
| lightgbm | 0.95 | 1.76 | 136.46% | 6045.6 |
| **vs Slot Mean** | -7.7% | — | — | — |
| **vs Seasonal Naive** | -28.8% | — | — | — |

### Feature Importance (LightGBM, top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | hour_of_day | 2285.0000 |
| 2 | lag_1h | 1584.0000 |
| 3 | day_of_week | 1482.0000 |
| 4 | lag_168h | 1383.0000 |
| 5 | lag_24h | 1371.0000 |
| 6 | rolling_mean_24h | 1324.0000 |
| 7 | zone_id | 937.0000 |
| 8 | is_weekend | 167.0000 |
| 9 | month | 0.0000 |

### Edge-Case Analysis

| Segment | Definition | N rows | Slot Mean MAE | LightGBM MAE | Δ (LightGBM vs Slot Mean) |
|---------|------------|--------|---------------|--------------|---------------------------|
| sparse_zones | Zones with mean demand < P10 | 79872 | 0.20 | 0.17 | -15.7% |
| dense_zones | Zones with mean demand > P90 | 82176 | 2.87 | 2.46 | -14.3% |
| late_night | Hours 0–5 | 205824 | 0.28 | 0.31 | +10.6% |
| peak_hours | Hours 7–9, 17–19 | 205824 | 1.40 | 1.28 | -8.6% |
| weekend | Saturday + Sunday (day_of_week >= 5) | 308736 | 1.13 | 0.97 | -14.3% |
| weekday | Monday–Friday (day_of_week < 5) | 514560 | 0.96 | 0.93 | -3.1% |
| zero_demand | Hours with trip_count == 0 | 457609 | 0.37 | 0.42 | +14.8% |

### Hardest Stations (by LightGBM MAE)

| ID | Name | MAE | Mean Demand |
|----|------|-----|-------------|
| 694810 | Broadway & W 58 St | 5.75 | 13.59 |
| 614005 | W 21 St & 6 Ave | 4.16 | 15.03 |
| 687604 | Central Park S & 6 Ave | 3.86 | 7.39 |
| 645005 | 8 Ave & W 31 St | 3.79 | 12.02 |
| 633101 | W 31 St & 7 Ave | 3.78 | 10.65 |

### Configuration

```yaml
train_start: 2024-02-01T00:00:00
train_end: 2024-03-01T00:00:00
val_end: 2024-03-16T00:00:00
test_end: 2024-04-01T00:00:00
n_zones: 2144
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
| slot_mean | `d0cd32bd4d1149858431fcb64dd9f4ab` |
| seasonal_naive | `5c362ca6e6f84034a16f142346462c00` |
| lightgbm | `c02dea070ef744e19b679b7cdbf32c83` |
