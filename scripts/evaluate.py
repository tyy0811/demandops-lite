"""Script entrypoint for evaluation.

Loads trained model artifacts from disk (does not retrain).
Run `make train` first to produce artifacts.
"""

from pathlib import Path

import yaml

from demandops.training.evaluate import evaluate_all
from demandops.training.train import load_trained_models


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())

    trained = load_trained_models(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
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
