"""Script entrypoint for evaluation."""

from pathlib import Path

import yaml

from demandops.training.evaluate import evaluate_all
from demandops.training.train import train_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())

    trained = train_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
        feature_schema_path=Path(config["artifacts"]["feature_schema_path"]),
    )

    report = evaluate_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        trained_models=trained,
        reports_dir=Path(config["artifacts"]["reports_dir"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
    )

    print("\n=== Model Comparison ===")
    for name, metrics in report["model_comparison"].items():
        print(f"{name}: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
              f"sMAPE={metrics['smape']:.2f}%")


if __name__ == "__main__":
    main()
