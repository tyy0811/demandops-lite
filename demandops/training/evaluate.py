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
            pl.col("zone_name").first().alias("zone_name"),
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
