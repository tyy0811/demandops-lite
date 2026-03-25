"""Run full benchmark: train → evaluate → markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from demandops.training.evaluate import evaluate_all
from demandops.training.train import train_all


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    print("Training all models...")
    trained = train_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
        feature_schema_path=Path(config["artifacts"]["feature_schema_path"]),
    )

    print("Evaluating on test set...")
    report = evaluate_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        trained_models=trained,
        reports_dir=Path(config["artifacts"]["reports_dir"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
    )

    _generate_markdown_report(report, config, trained)


# Dataset display names for benchmark reports
DATASET_LABELS: dict[str, str] = {
    "taxi": "NYC Taxi Demand Prediction",
    "tfl": "London Cycle Hire Demand Prediction",
}

ENTITY_LABELS: dict[str, str] = {
    "taxi": "pickup zone",
    "tfl": "docking station",
}


def _generate_markdown_report(report: dict, config: dict, trained: dict) -> None:
    zone_universe = json.loads(Path(config["artifacts"]["zone_universe_path"]).read_text())

    adapter_name = config.get("dataset", {}).get("adapter", "taxi")
    dataset_label = DATASET_LABELS.get(adapter_name, adapter_name)
    entity_label = ENTITY_LABELS.get(adapter_name, "zone")

    # Compute grid size from features parquet
    features_path = Path(config["data"]["processed_dir"]) / "features.parquet"
    features_df = pl.read_parquet(features_path)
    n_features_rows = len(features_df)

    split_cfg = config["split"]

    lines = [
        f"## Benchmark Results — {dataset_label}\n",
        f"**Target:** Hourly trip count per {entity_label}",
        f"**Entities:** {zone_universe['n_zones']} (from zone_universe.json)",
        (
            f"**Grid:** {n_features_rows:,} rows "
            f"({zone_universe['n_zones']} {entity_label}s × hourly)"
        ),
        (
            f"**Train:** [{split_cfg['train_start'][:10]}, {split_cfg['train_end'][:10]}) | "
            f"**Val:** [{split_cfg['train_end'][:10]}, {split_cfg['val_end'][:10]}) | "
            f"**Test:** [{split_cfg['val_end'][:10]}, {split_cfg['test_end'][:10]})"
        ),
        "**Features:** 9 (temporal + lag)",
        "",
    ]

    lgbm = report["model_comparison"].get("lightgbm", {})
    if lgbm:
        lines.append(
            f"**Negative prediction handling:** LightGBM predictions clipped to zero "
            f"({lgbm.get('n_clipped_to_zero', 0)} predictions, "
            f"{lgbm.get('pct_clipped', 0):.1f}%)"
        )
        lines.append("")

    # --- Model Comparison ---
    lines.append("### Model Comparison\n")
    lines.append("| Model | MAE | RMSE | sMAPE | Latency (ms) |")
    lines.append("|-------|-----|------|-------|-------------|")

    for name in ["slot_mean", "seasonal_naive", "lightgbm"]:
        m = report["model_comparison"].get(name, {})
        lines.append(
            f"| {name} | {m.get('mae', 0):.2f} | {m.get('rmse', 0):.2f} | "
            f"{m.get('smape', 0):.2f}% | {m.get('latency_ms', 0):.1f} |"
        )

    # Delta rows
    sm = report["model_comparison"].get("slot_mean", {})
    sn = report["model_comparison"].get("seasonal_naive", {})
    lg = report["model_comparison"].get("lightgbm", {})

    if sm and lg and sm.get("mae", 0) > 0:
        d = (lg["mae"] - sm["mae"]) / sm["mae"] * 100
        lines.append(f"| **vs Slot Mean** | {d:+.1f}% | — | — | — |")
    if sn and lg and sn.get("mae", 0) > 0:
        d = (lg["mae"] - sn["mae"]) / sn["mae"] * 100
        lines.append(f"| **vs Seasonal Naive** | {d:+.1f}% | — | — | — |")

    # --- Feature Importance ---
    if report.get("feature_importance"):
        lines.append("\n### Feature Importance (LightGBM, top 10)\n")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for i, (feat, imp) in enumerate(report["feature_importance"], 1):
            lines.append(f"| {i} | {feat} | {imp:.4f} |")

    # --- Edge-Case Analysis ---
    segment_definitions = {
        "sparse_zones": "Zones with mean demand < P10",
        "dense_zones": "Zones with mean demand > P90",
        "late_night": "Hours 0–5",
        "peak_hours": "Hours 7–9, 17–19",
        "weekend": "Saturday + Sunday (day_of_week >= 5)",
        "weekday": "Monday–Friday (day_of_week < 5)",
        "zero_demand": "Hours with trip_count == 0",
    }

    if report.get("edge_cases"):
        lines.append("\n### Edge-Case Analysis\n")
        lines.append(
            "| Segment | Definition | N rows | Slot Mean MAE | "
            "LightGBM MAE | Δ (LightGBM vs Slot Mean) |"
        )
        lines.append(
            "|---------|------------|--------|---------------|"
            "--------------|---------------------------|"
        )
        for seg, data in report["edge_cases"].items():
            defn = segment_definitions.get(seg, "—")
            sm_m = data.get("slot_mean_mae")
            lg_m = data.get("lightgbm_mae")
            sm_str = f"{sm_m:.2f}" if isinstance(sm_m, float) else "—"
            lg_str = f"{lg_m:.2f}" if isinstance(lg_m, float) else "—"
            if isinstance(sm_m, float) and isinstance(lg_m, float) and sm_m > 0:
                delta = (lg_m - sm_m) / sm_m * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "—"
            lines.append(
                f"| {seg} | {defn} | {data['n_rows']} | {sm_str} | {lg_str} | {delta_str} |"
            )

    # --- Hardest Zones ---
    if report.get("per_zone_top5"):
        lines.append(f"\n### Hardest {entity_label.title()}s (by LightGBM MAE)\n")
        lines.append("| ID | Name | MAE | Mean Demand |")
        lines.append("|----|------|-----|-------------|")
        for entry in report["per_zone_top5"]:
            lines.append(
                f"| {entry['zone_id']} | {entry.get('zone_name', 'Unknown')} | "
                f"{entry['mae']:.2f} | {entry['mean_demand']:.2f} |"
            )

    # --- Configuration ---
    lines.append("\n### Configuration\n")
    lines.append("```yaml")
    split_cfg = config["split"]
    lines.append(f"train_start: {split_cfg['train_start']}")
    lines.append(f"train_end: {split_cfg['train_end']}")
    lines.append(f"val_end: {split_cfg['val_end']}")
    lines.append(f"test_end: {split_cfg['test_end']}")
    lines.append(f"n_zones: {zone_universe['n_zones']}")
    lgbm_cfg = config["models"].get("lightgbm", {})
    for k, v in lgbm_cfg.items():
        if k != "name":
            lines.append(f"lightgbm.{k}: {v}")
    lines.append("```")

    # --- MLflow Run IDs ---
    lines.append("\n### MLflow Run IDs\n")
    lines.append("| Model | Run ID |")
    lines.append("|-------|--------|")
    for name in ["slot_mean", "seasonal_naive", "lightgbm"]:
        info = trained.get(name, {})
        run_id = info.get("run_id", "—")
        lines.append(f"| {name} | `{run_id}` |")

    # Dataset-specific report path
    if adapter_name == "taxi":
        report_path = Path("docs/benchmark_report.md")
    else:
        report_path = Path(f"docs/benchmark_report_{adapter_name}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nBenchmark report saved to {report_path}")


if __name__ == "__main__":
    main()
