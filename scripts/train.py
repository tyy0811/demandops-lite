"""Script entrypoint for model training."""

import argparse
from pathlib import Path

import yaml

from demandops.training.train import train_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    results = train_all(
        features_path=Path(config["data"]["processed_dir"]) / "features.parquet",
        config=config,
        models_dir=Path(config["artifacts"]["models_dir"]),
        feature_schema_path=Path(config["artifacts"]["feature_schema_path"]),
    )
    for name, info in results.items():
        print(f"{name}: val_mae={info['val_mae']:.4f}, run_id={info['run_id']}")


if __name__ == "__main__":
    main()
