## Benchmark Results — London Cycle Hire Demand Prediction

**Target:** Hourly trip count per docking station
**Entities:** 802 (from zone_universe.json)
**Grid:** 1,154,880 rows (802 docking stations × hourly)
**Train:** [2024-01-01, 2024-02-01) | **Val:** [2024-02-01, 2024-02-15) | **Test:** [2024-02-15, 2024-03-01)
**Features:** 9 (temporal + lag)

**Negative prediction handling:** LightGBM predictions clipped to zero (2109 predictions, 0.7%)

### Model Comparison

| Model | MAE | RMSE | sMAPE | Latency (ms) |
|-------|-----|------|-------|-------------|
| slot_mean | 0.75 | 1.31 | 124.31% | 364.1 |
| seasonal_naive | 0.89 | 1.63 | 124.46% | 0.3 |
| lightgbm | 0.77 | 1.28 | 138.19% | 2139.9 |
| **vs Slot Mean** | +2.0% | — | — | — |
| **vs Seasonal Naive** | -13.7% | — | — | — |

### Feature Importance (LightGBM, top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | lag_1h | 1490.0000 |
| 2 | lag_168h | 1393.0000 |
| 3 | hour_of_day | 1370.0000 |
| 4 | lag_24h | 1239.0000 |
| 5 | rolling_mean_24h | 964.0000 |
| 6 | day_of_week | 774.0000 |
| 7 | zone_id | 278.0000 |
| 8 | is_weekend | 62.0000 |
| 9 | month | 0.0000 |

### Edge-Case Analysis

| Segment | Definition | N rows | Slot Mean MAE | LightGBM MAE | Δ (LightGBM vs Slot Mean) |
|---------|------------|--------|---------------|--------------|---------------------------|
| sparse_zones | Zones with mean demand < P10 | 28800 | 0.36 | 0.37 | +3.3% |
| dense_zones | Zones with mean demand > P90 | 28800 | 1.29 | 1.28 | -0.5% |
| late_night | Hours 0–5 | 72180 | 0.16 | 0.17 | +9.0% |
| peak_hours | Hours 7–9, 17–19 | 72180 | 1.17 | 1.20 | +2.7% |
| weekend | Saturday + Sunday (day_of_week >= 5) | 76992 | 0.69 | 0.71 | +2.3% |
| weekday | Monday–Friday (day_of_week < 5) | 211728 | 0.77 | 0.79 | +1.8% |
| zero_demand | Hours with trip_count == 0 | 163684 | 0.39 | 0.49 | +26.5% |

### Hardest Docking Stations (by LightGBM MAE)

| ID | Name | MAE | Mean Demand |
|----|------|-----|-------------|
| 2696 | Waterloo Station 1, Waterloo | 2.49 | 4.52 |
| 1072 | Waterloo Station 3, Waterloo | 2.47 | 6.48 |
| 1075 | Hyde Park Corner, Hyde Park | 2.25 | 3.54 |
| 2692 | Waterloo Station 2, Waterloo | 1.91 | 3.17 |
| 1132 | Albert Gate, Hyde Park | 1.76 | 2.57 |

### Configuration

```yaml
train_start: 2024-01-01T00:00:00
train_end: 2024-02-01T00:00:00
val_end: 2024-02-15T00:00:00
test_end: 2024-03-01T00:00:00
n_zones: 802
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
| slot_mean | `9546afd832db473ea3db80351ef5e736` |
| seasonal_naive | `79bfb32e64c2470aa11dc25bf988ab5a` |
| lightgbm | `6cdc9aeeb1a4464785e0aef1891f6ab5` |
