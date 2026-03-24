"""Tests for training and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from demandops.training.evaluate import evaluate_all
from demandops.training.train import load_trained_models, train_all


@pytest.fixture
def training_config(split_config: dict) -> dict:
    """Minimal config for training tests."""
    return {
        **split_config,
        "models": {
            "slot_mean": {"name": "slot_mean"},
            "seasonal_naive": {"name": "seasonal_naive"},
            "lightgbm": {
                "name": "lightgbm",
                "n_estimators": 5,
                "num_threads": 1,
                "random_state": 42,
                "verbose": -1,
            },
        },
        "mlflow": {
            "tracking_uri": "file:./mlruns",
            "experiment_name": "test-run",
        },
    }


@pytest.fixture
def features_parquet(tmp_path: Path, features_df: pl.DataFrame) -> Path:
    path = tmp_path / "features.parquet"
    features_df.write_parquet(path)
    return path


class TestTrainAll:

    def test_trains_all_models(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"

        results = train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        assert set(results.keys()) == {"slot_mean", "seasonal_naive", "lightgbm"}
        for name, info in results.items():
            assert "model" in info
            assert "val_mae" in info
            assert info["val_mae"] >= 0

    def test_saves_lightgbm_artifact(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"

        train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        assert (models_dir / "lightgbm.joblib").exists()

    def test_saves_feature_schema(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"

        train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        assert schema_path.exists()
        schema = json.loads(schema_path.read_text())
        assert "columns" in schema
        assert "target" in schema


class TestLoadTrainedModels:

    def test_loads_from_artifacts(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        """load_trained_models() loads LightGBM from disk, not by retraining."""
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"

        # First train to produce artifacts
        train_results = train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        # Now load from artifacts
        loaded = load_trained_models(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
        )

        assert set(loaded.keys()) == {"slot_mean", "seasonal_naive", "lightgbm"}

        # LightGBM predictions should match the trained model
        import numpy as np
        rng = np.random.RandomState(99)
        X_test = rng.rand(5, 9)
        trained_preds = train_results["lightgbm"]["model"].predict(X_test)
        loaded_preds = loaded["lightgbm"]["model"].predict(X_test)
        np.testing.assert_array_almost_equal(trained_preds, loaded_preds)

    def test_missing_artifact_raises(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        """load_trained_models() raises if LightGBM artifact doesn't exist."""
        models_dir = tmp_path / "empty_models"
        models_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="make train"):
            load_trained_models(
                features_path=features_parquet,
                config=training_config,
                models_dir=models_dir,
            )


class TestEvaluateAll:

    def test_produces_report_with_all_models(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"
        reports_dir = tmp_path / "reports"
        zu_path = tmp_path / "zone_universe.json"
        zu_path.write_text(json.dumps({"zone_ids": [1, 2, 3], "n_zones": 3}))

        trained = train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        report = evaluate_all(
            features_path=features_parquet,
            config=training_config,
            trained_models=trained,
            reports_dir=reports_dir,
            zone_universe_path=zu_path,
        )

        assert "model_comparison" in report
        assert set(report["model_comparison"].keys()) == {
            "slot_mean", "seasonal_naive", "lightgbm",
        }
        for metrics in report["model_comparison"].values():
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "smape" in metrics

    def test_per_zone_includes_zone_name(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"
        reports_dir = tmp_path / "reports"
        zu_path = tmp_path / "zone_universe.json"
        zu_path.write_text(json.dumps({"zone_ids": [1, 2, 3], "n_zones": 3}))

        trained = train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        report = evaluate_all(
            features_path=features_parquet,
            config=training_config,
            trained_models=trained,
            reports_dir=reports_dir,
            zone_universe_path=zu_path,
        )

        per_zone = report["per_zone_top5"]
        assert len(per_zone) > 0
        for entry in per_zone:
            assert "zone_name" in entry, "per-zone report missing zone_name"
            assert "zone_id" in entry
            assert "mae" in entry

    def test_report_saved_to_disk(
        self, features_parquet: Path, training_config: dict, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        schema_path = tmp_path / "feature_schema.json"
        reports_dir = tmp_path / "reports"
        zu_path = tmp_path / "zone_universe.json"
        zu_path.write_text(json.dumps({"zone_ids": [1, 2, 3], "n_zones": 3}))

        trained = train_all(
            features_path=features_parquet,
            config=training_config,
            models_dir=models_dir,
            feature_schema_path=schema_path,
        )

        evaluate_all(
            features_path=features_parquet,
            config=training_config,
            trained_models=trained,
            reports_dir=reports_dir,
            zone_universe_path=zu_path,
        )

        assert (reports_dir / "eval_results.json").exists()
        saved = json.loads((reports_dir / "eval_results.json").read_text())
        assert "model_comparison" in saved
