"""Training pipeline: load features, split, train models, save artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import structlog

from demandops.data.splits import split_from_config
from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN
from demandops.models.registry import create_model

logger = structlog.get_logger()


def train_model(
    features_path: Path,
    config: dict,
    model_name: str,
    models_dir: Path,
    feature_schema_path: Path,
) -> dict[str, Any]:
    """Train a single model and save artifacts."""
    df = pl.read_parquet(features_path)
    train, val, _ = split_from_config(df, config)

    logger.info("split_sizes", train=len(train), val=len(val), model=model_name)

    X_train = train.select(FEATURE_COLUMNS).to_numpy()
    y_train = train[TARGET_COLUMN].to_numpy().astype(float)
    X_val = val.select(FEATURE_COLUMNS).to_numpy()
    y_val = val[TARGET_COLUMN].to_numpy().astype(float)

    model_config = config["models"].get(model_name, {})
    model_params = {k: v for k, v in model_config.items() if k != "name"}
    model = create_model(model_name, **model_params)

    if model_name == "lightgbm":
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    # Save feature schema (same for all models)
    _save_feature_schema(feature_schema_path)

    # Save model artifact (fix #4: joblib for LightGBM)
    models_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "lightgbm":
        model_path = models_dir / f"{model_name}.joblib"
        model.save(model_path)
        logger.info("model_saved", path=str(model_path))

    # MLflow logging
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(model.get_params())
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))

        val_preds = model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        mlflow.log_metric("val_mae", val_mae)
        logger.info("val_mae", model=model_name, mae=val_mae)

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "val_mae": val_mae,
            "model": model,
        }


def _save_feature_schema(path: Path) -> None:
    schema = {
        "columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "categorical": ["zone_id"],
        "numerical": [c for c in FEATURE_COLUMNS if c != "zone_id"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2))


def train_all(
    features_path: Path,
    config: dict,
    models_dir: Path,
    feature_schema_path: Path,
) -> dict[str, Any]:
    """Train all models defined in config."""
    results = {}
    for model_name in config["models"]:
        result = train_model(
            features_path=features_path,
            config=config,
            model_name=model_name,
            models_dir=models_dir,
            feature_schema_path=feature_schema_path,
        )
        results[model_name] = result
    return results
