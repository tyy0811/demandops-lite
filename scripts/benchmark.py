"""Run full benchmark: train → evaluate → markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from demandops.training.evaluate import evaluate_all
from demandops.training.train import train_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())

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

    _generate_markdown_report(report, config)


def _generate_markdown_report(report: dict, config: dict) -> None:
    zone_universe = json.loads(
        Path(config["artifacts"]["zone_universe_path"]).read_text()
    )

    lines = [
        "## Benchmark Results — NYC Taxi Demand Prediction\n",
        f"**Dataset:** NYC TLC Yellow Taxi, Jan–Feb 2024 (Dec 2023 for warm-up)",
        f"**Target:** Hourly trip count per pickup zone",
        f"**Zones:** {zone_universe['n_zones']} (from zone_universe.json)",
        f"**Train:** [2024-01-01, 2024-02-01) | **Val:** [2024-02-01, 2024-02-15) | **Test:** [2024-02-15, 2024-03-01)",
        f"**Features:** 9 (temporal + lag)",
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

    if report.get("feature_importance"):
        lines.append("\n### Feature Importance (LightGBM, top 10)\n")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for i, (feat, imp) in enumerate(report["feature_importance"], 1):
            lines.append(f"| {i} | {feat} | {imp:.4f} |")

    if report.get("edge_cases"):
        lines.append("\n### Edge-Case Analysis\n")
        lines.append("| Segment | N rows | Slot Mean MAE | LightGBM MAE |")
        lines.append("|---------|--------|-------------|-------------|")
        for seg, data in report["edge_cases"].items():
            sm_m = data.get("slot_mean_mae", "—")
            lg_m = data.get("lightgbm_mae", "—")
            if isinstance(sm_m, float):
                sm_m = f"{sm_m:.2f}"
            if isinstance(lg_m, float):
                lg_m = f"{lg_m:.2f}"
            lines.append(f"| {seg} | {data['n_rows']} | {sm_m} | {lg_m} |")

    if report.get("per_zone_top5"):
        lines.append("\n### Hardest Zones (by LightGBM MAE)\n")
        lines.append("| Zone ID | Zone Name | MAE | Mean Demand |")
        lines.append("|---------|-----------|-----|-------------|")
        for entry in report["per_zone_top5"]:
            lines.append(
                f"| {entry['zone_id']} | {entry.get('zone_name', 'Unknown')} | "
                f"{entry['mae']:.2f} | {entry['mean_demand']:.2f} |"
            )

    report_path = Path("docs/benchmark_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nBenchmark report saved to {report_path}")


if __name__ == "__main__":
    main()
