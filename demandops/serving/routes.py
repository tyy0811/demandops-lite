"""API routes: /predict, /health, /metrics."""

from __future__ import annotations

import time
import uuid

import numpy as np
import structlog
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from prometheus_client import generate_latest
from starlette.responses import Response

from demandops.serving.metrics import (
    ERROR_COUNT,
    HISTORY_LOADED,
    MODEL_LOADED,
    PREDICTION_COUNT,
    PREDICTION_VALUE,
    REJECTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from demandops.security.auth import requires_auth
from demandops.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionMetadata,
)

logger = structlog.get_logger()

router = APIRouter()


def configure(
    app: FastAPI,
    feature_service,
    model,
    model_name: str,
    start_time: float,
    model_artifact_loaded: bool = True,
    model_objective: str = "regression",
    model_version: str = "v1",
):
    """Store dependencies on app.state (fix #8: no module-level globals)."""
    app.state.feature_service = feature_service
    app.state.model = model
    app.state.model_name = model_name
    app.state.start_time = start_time
    app.state.model_artifact_loaded = model_artifact_loaded
    app.state.model_objective = model_objective
    app.state.model_version = model_version


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request, client: dict = Depends(requires_auth)):
    request_id = str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    start = time.perf_counter()
    svc = request.app.state.feature_service
    model = request.app.state.model
    model_name = request.app.state.model_name
    model_version = request.app.state.model_version

    try:
        result = svc.get_features(body.zone_id, body.hour_ts)

        if not result.supported:
            reason = (
                "unsupported_zone"
                if result.warnings and "zone universe" in result.warnings[0]
                else "unsupported_timestamp"
            )
            REJECTION_COUNT.labels(reason=reason).inc()
            REQUEST_COUNT.labels(endpoint="/predict", status="422").inc()
            REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - start)
            raise HTTPException(
                status_code=422,
                detail={
                    "detail": result.warnings[0] if result.warnings else "Unsupported",
                    "supported_start": svc.supported_start.isoformat(),
                    "supported_end": svc.supported_end.isoformat(),
                    "n_supported_zones": svc.n_supported_zones,
                },
            )

        features = result.features
        X = np.array([[features[col] for col in features]], dtype=float)

        predicted_count = float(model.predict(X)[0])
        latency_ms = (time.perf_counter() - start) * 1000

        PREDICTION_COUNT.inc()
        PREDICTION_VALUE.observe(predicted_count)
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - start)

        # Feed drift accumulator
        drift_detector = getattr(request.app.state, "drift_detector", None)
        if drift_detector is not None:
            drift_detector.accumulator.add([features[col] for col in features])

        # Log to quality tracker
        quality_tracker = getattr(request.app.state, "quality_tracker", None)
        if quality_tracker is not None:
            prediction_id = quality_tracker.log_prediction(
                zone_id=body.zone_id,
                hour_ts=body.hour_ts.isoformat(),
                predicted_value=predicted_count,
            )

        logger.info(
            "prediction",
            zone_id=body.zone_id,
            predicted_count=round(predicted_count, 4),
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )

        return PredictResponse(
            prediction_id=prediction_id,
            zone_id=body.zone_id,
            zone_name=svc.get_zone_name(body.zone_id),
            hour_ts=body.hour_ts,
            predicted_count=predicted_count,
            model_name=model_name,
            model_version=model_version,
            metadata=PredictionMetadata(
                latency_ms=latency_ms,
                request_id=request_id,
                features_used=features,
                input_warnings=result.warnings,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - start)
        logger.error("prediction_error", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    body: BatchPredictRequest, request: Request, client: dict = Depends(requires_auth)
):
    start = time.perf_counter()
    svc = request.app.state.feature_service
    model = request.app.state.model
    model_name = request.app.state.model_name
    model_version = request.app.state.model_version

    # Per-key batch size enforcement
    if len(body.requests) > client["max_batch_size"]:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(body.requests)} exceeds limit {client['max_batch_size']}",
        )

    try:
        # Phase 1: Collect features for all requests (all-or-nothing validation)
        feature_dicts = []
        zone_names = []
        for req in body.requests:
            result = svc.get_features(req.zone_id, req.hour_ts)

            if not result.supported:
                reason = (
                    "unsupported_zone"
                    if result.warnings and "zone universe" in result.warnings[0]
                    else "unsupported_timestamp"
                )
                REJECTION_COUNT.labels(reason=reason).inc()
                REQUEST_COUNT.labels(endpoint="/predict/batch", status="422").inc()
                REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(
                    time.perf_counter() - start
                )
                raise HTTPException(
                    status_code=422,
                    detail={
                        "detail": result.warnings[0] if result.warnings else "Unsupported",
                        "supported_start": svc.supported_start.isoformat(),
                        "supported_end": svc.supported_end.isoformat(),
                        "n_supported_zones": svc.n_supported_zones,
                    },
                )

            feature_dicts.append(result.features)
            zone_names.append(svc.get_zone_name(req.zone_id))

        # Phase 2: Vectorized prediction
        X = np.array(
            [[f[col] for col in f] for f in feature_dicts],
            dtype=float,
        )
        raw_preds = model.predict(X)

        # Feed drift accumulator with all feature vectors
        drift_detector = getattr(request.app.state, "drift_detector", None)
        if drift_detector is not None:
            drift_detector.accumulator.add_batch([[f[col] for col in f] for f in feature_dicts])

        # Phase 3: Build responses
        quality_tracker = getattr(request.app.state, "quality_tracker", None)
        predictions = []
        for i, req in enumerate(body.requests):
            predicted_count = float(raw_preds[i])
            PREDICTION_COUNT.inc()
            PREDICTION_VALUE.observe(predicted_count)

            # Log to quality tracker
            if quality_tracker is not None:
                pred_id = quality_tracker.log_prediction(
                    zone_id=req.zone_id,
                    hour_ts=req.hour_ts.isoformat(),
                    predicted_value=predicted_count,
                )
            else:
                pred_id = str(uuid.uuid4())

            predictions.append(
                PredictResponse(
                    prediction_id=pred_id,
                    zone_id=req.zone_id,
                    zone_name=zone_names[i],
                    hour_ts=req.hour_ts,
                    predicted_count=predicted_count,
                    model_name=model_name,
                    model_version=model_version,
                    metadata=PredictionMetadata(
                        latency_ms=0.0,  # Individual latency not meaningful in batch
                        request_id="batch",
                        features_used=feature_dicts[i],
                    ),
                )
            )

        latency_ms = (time.perf_counter() - start) * 1000
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(time.perf_counter() - start)

        logger.info(
            "batch_prediction",
            count=len(predictions),
            latency_ms=round(latency_ms, 2),
        )

        return BatchPredictResponse(
            predictions=predictions,
            prediction_count=len(predictions),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="500").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(time.perf_counter() - start)
        logger.error("batch_prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    svc = request.app.state.feature_service
    model_name = request.app.state.model_name
    start_time = request.app.state.start_time

    model_loaded = getattr(request.app.state, "model_artifact_loaded", False)
    history_loaded = svc is not None

    MODEL_LOADED.set(1 if model_loaded else 0)
    HISTORY_LOADED.set(1 if history_loaded else 0)
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()

    model_objective = getattr(request.app.state, "model_objective", "regression")
    model_version = getattr(request.app.state, "model_version", "v1")

    return HealthResponse(
        status="healthy" if model_loaded and history_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name or "none",
        model_objective=model_objective,
        model_version=model_version,
        history_loaded=history_loaded,
        supported_start=svc.supported_start if history_loaded else None,
        supported_end=svc.supported_end if history_loaded else None,
        n_supported_zones=svc.n_supported_zones if history_loaded else 0,
        history_rows=len(svc.history) if history_loaded else 0,
        uptime_seconds=time.time() - start_time if start_time else 0,
    )


@router.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
