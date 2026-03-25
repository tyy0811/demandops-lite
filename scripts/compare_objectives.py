"""Compare LightGBM regression vs Poisson objective.

Trains two models with identical hyperparameters except objective.
Logs both runs to MLflow. Prints comparison table.

Usage: python scripts/compare_objectives.py
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import yaml

from demandops.data.splits import split_from_config
from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN
from demandops.models.registry import create_model


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    features_path = Path(config["data"]["processed_dir"]) / "features.parquet"

    df = pl.read_parquet(features_path)
    train, val, test = split_from_config(df, config)

    X_train = train.select(FEATURE_COLUMNS).to_numpy()
    y_train = train[TARGET_COLUMN].to_numpy().astype(float)
    X_val = val.select(FEATURE_COLUMNS).to_numpy()
    y_val = val[TARGET_COLUMN].to_numpy().astype(float)
    X_test = test.select(FEATURE_COLUMNS).to_numpy()
    y_test = test[TARGET_COLUMN].to_numpy().astype(float)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("objective-comparison")

    results = []

    for objective in ["regression", "poisson"]:
        print(f"\nTraining with objective={objective}...")
        start = time.perf_counter()

        model_params = {k: v for k, v in config["models"]["lightgbm"].items() if k != "name"}
        model_params["objective"] = objective

        model = create_model("lightgbm", **model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        elapsed = time.perf_counter() - start

        # Evaluate on test set
        preds = model.predict(X_test)
        raw_preds = model.predict_raw(X_test)
        test_mae = float(np.mean(np.abs(y_test - preds)))
        neg_count = int((raw_preds < 0).sum())

        result = {
            "objective": objective,
            "mae": test_mae,
            "neg_preds": neg_count,
            "neg_pct": round(100 * neg_count / len(raw_preds), 2),
            "mean_pred": round(float(preds.mean()), 4),
            "std_pred": round(float(preds.std()), 4),
            "best_iter": model.get_params().get("best_iteration", "N/A"),
            "time_sec": round(elapsed, 2),
        }
        results.append(result)

        # Log to MLflow
        with mlflow.start_run(run_name=f"lightgbm_{objective}"):
            mlflow.log_params(model_params)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("negative_predictions", neg_count)
            mlflow.log_metric("training_time_sec", elapsed)

    # Print comparison
    print("\n" + "=" * 70)
    print("OBJECTIVE COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Objective':<12} {'MAE':<8} {'Neg Preds':<11} {'Best Iter':<11} {'Time':>6}")
    print("-" * 54)
    for r in results:
        print(
            f"{r['objective']:<12} "
            f"{r['mae']:<8.4f} "
            f"{r['neg_preds']:<5} ({r['neg_pct']:.1f}%) "
            f"{str(r['best_iter']):<11} "
            f"{r['time_sec']:>5.1f}s"
        )

    # Determine winner
    reg = results[0]
    poi = results[1]
    if poi["mae"] < reg["mae"]:
        delta = (reg["mae"] - poi["mae"]) / reg["mae"] * 100
        print(f"\nPoisson wins by {delta:.1f}% MAE improvement.")
        if poi["neg_preds"] == 0:
            print("Poisson also eliminates negative predictions entirely.")
    elif poi["mae"] == reg["mae"]:
        print("\nTied on MAE.")
        if poi["neg_preds"] < reg["neg_preds"]:
            print("Poisson eliminates negative predictions — qualitative improvement.")
    else:
        delta = (poi["mae"] - reg["mae"]) / reg["mae"] * 100
        print(f"\nRegression wins by {delta:.1f}% MAE.")
        print(f"Negative predictions in regression: {reg['neg_preds']} (clipped in serving).")

    print("\nBoth runs logged to MLflow experiment 'objective-comparison'.")
    print("See DECISIONS.md for the write-up.\n")


if __name__ == "__main__":
    main()
