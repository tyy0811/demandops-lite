"""FastAPI application factory."""

from __future__ import annotations

import time
from pathlib import Path

import structlog
import yaml
from fastapi import FastAPI

from demandops.db import get_db
from demandops.models.registry import create_model
from demandops.monitoring.drift_detector import DriftDetector
from demandops.monitoring.quality_tracker import QualityTracker
from demandops.security.auth import RateLimiter
from demandops.serving.feature_service import FeatureService
from demandops.serving.middleware import RequestLoggingMiddleware
from demandops.serving.monitoring_routes import monitoring_router
from demandops.serving.routes import configure, router

logger = structlog.get_logger()


def create_app(config_path: str = "configs/default.yaml") -> FastAPI:
    config = yaml.safe_load(Path(config_path).read_text())
    serving_cfg = config["serving"]

    app = FastAPI(
        title="demandops-lite",
        description="Hourly taxi demand prediction API",
        version="0.1.0",
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router)
    app.include_router(monitoring_router)

    @app.on_event("startup")
    async def startup():
        start_time = time.time()
        model_name = serving_cfg["model_name"]
        feature_service = None
        model = None
        model_artifact_loaded = False

        # Initialize shared database and rate limiter
        db_path = config.get("db", {}).get("path", "data/demandops.db")
        db = get_db(db_path)
        app.state.db = db
        app.state.rate_limiter = RateLimiter()

        # Load feature service — graceful degradation on missing artifacts
        try:
            feature_service = FeatureService(
                history_path=Path(serving_cfg["history_path"]),
                schema_path=Path(serving_cfg["feature_schema_path"]),
                zone_universe_path=Path(serving_cfg["zone_universe_path"]),
                config=config,
            )
        except Exception as e:
            logger.error("feature_service_failed", error=str(e))

        # Load model (fix #4: joblib for LightGBM)
        model_config = config["models"].get(model_name, {})
        model_params = {k: v for k, v in model_config.items() if k != "name"}

        try:
            if model_name == "lightgbm":
                model_path = Path(config["artifacts"]["models_dir"]) / f"{model_name}.joblib"
                if model_path.exists():
                    model = create_model(model_name, **model_params)
                    model.load(model_path)
                    model_artifact_loaded = True
                    logger.info("model_loaded", path=str(model_path))
                else:
                    logger.warning("model_file_not_found", path=str(model_path))
            else:
                model = create_model(model_name, **model_params)
                model_artifact_loaded = True
        except Exception as e:
            logger.error("model_load_failed", error=str(e))

        model_objective = model_params.get("objective", "regression")
        model_version = f"{model_name}-{model_objective}"

        configure(
            app,
            feature_service,
            model,
            model_name,
            start_time,
            model_artifact_loaded=model_artifact_loaded,
            model_objective=model_objective,
            model_version=model_version,
        )

        # Initialize drift detector (graceful degradation if reference missing)
        ref_path = Path(
            config["artifacts"].get(
                "reference_distributions_path",
                "artifacts/reference_distributions.json",
            )
        )
        if ref_path.exists():
            monitoring_cfg = config.get("monitoring", {}).get("drift", {})
            app.state.drift_detector = DriftDetector(
                ref_path,
                maxlen=monitoring_cfg.get("maxlen", 1000),
                min_samples=monitoring_cfg.get("min_samples", 100),
            )
            logger.info("drift_detector_loaded", path=str(ref_path))
        else:
            app.state.drift_detector = None
            logger.warning("drift_detector_skipped", path=str(ref_path))

        # Initialize quality tracker
        app.state.quality_tracker = QualityTracker(db)

        # Expose monitoring config for routes
        quality_cfg = config.get("monitoring", {}).get("quality", {})
        app.state.monitoring_config = {
            "mae_threshold": 3.20,  # From regression gate
            "mae_alert_margin": quality_cfg.get("mae_alert_margin", 1.2),
        }

        logger.info(
            "app_started",
            model=model_name,
            objective=model_objective,
            artifact_loaded=model_artifact_loaded,
            history_loaded=feature_service is not None,
        )

    return app


app = create_app()
