"""Monitoring API routes: /monitoring/drift, /monitoring/quality, /monitoring/actuals."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from demandops.security.auth import get_usage, requires_auth
from demandops.serving.metrics import (
    DRIFT_ALERT,
    DRIFT_CORRELATION_SHIFT,
    DRIFT_KS_PVALUE,
    DRIFT_PSI,
    QUALITY_ALERT,
    QUALITY_MAE,
    QUALITY_RMSE,
    QUALITY_SMAPE,
)
from demandops.serving.schemas import ActualsRequest, ActualsResponse

monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_VERDICT_TO_GAUGE = {"ok": 0, "warning": 0.5, "alert": 1}


@monitoring_router.get("/drift")
async def drift_status(request: Request):
    """Return current drift status per feature. No auth required."""
    detector = request.app.state.drift_detector
    if detector is None:
        return {"status": "disabled", "reason": "reference distributions not loaded"}
    result = detector.compute_drift()

    # Update Prometheus gauges
    if "features" in result:
        for feature_name, metrics in result["features"].items():
            DRIFT_PSI.labels(feature=feature_name).set(metrics["psi"])
            DRIFT_KS_PVALUE.labels(feature=feature_name).set(metrics["ks_pvalue"])
            DRIFT_ALERT.labels(feature=feature_name).set(
                _VERDICT_TO_GAUGE.get(metrics["verdict"], 0)
            )
    if "correlation_shift" in result:
        DRIFT_CORRELATION_SHIFT.set(result["correlation_shift"])

    return result


@monitoring_router.get("/quality")
async def quality_status(request: Request, window: str = "7d"):
    """Return quality metrics over the specified window. No auth required.

    Includes drift-quality correlation when MAE exceeds threshold.
    """
    tracker = request.app.state.quality_tracker
    try:
        result = tracker.compute_quality(window=window)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Update Prometheus gauges
    quality_degraded = False
    if result.get("status") == "ok":
        QUALITY_MAE.set(result["mae"])
        QUALITY_RMSE.set(result["rmse"])
        QUALITY_SMAPE.set(result["smape"])

        # Drift-quality correlation: if quality is degraded, include drift status
        monitoring_cfg = getattr(request.app.state, "monitoring_config", {})
        mae_threshold = monitoring_cfg.get("mae_threshold", 3.20)
        margin = monitoring_cfg.get("mae_alert_margin", 1.2)
        if result.get("mae", 0) > mae_threshold * margin:
            quality_degraded = True
            detector = request.app.state.drift_detector
            if detector is not None:
                drift = detector.compute_drift()
                result["drift_correlation"] = {
                    "drift_status": drift.get("status"),
                    "note": "Quality degradation detected alongside drift — may indicate retraining needed",
                }

    QUALITY_ALERT.set(1 if quality_degraded else 0)

    return result


@monitoring_router.post("/actuals", response_model=ActualsResponse)
async def submit_actuals(
    body: ActualsRequest,
    request: Request,
    client: dict = Depends(requires_auth),
):
    """Submit ground truth actuals to match against logged predictions."""
    tracker = request.app.state.quality_tracker
    actuals = [a.model_dump() for a in body.actuals]
    result = tracker.submit_actuals(actuals)
    return ActualsResponse(**result)


@monitoring_router.get("/usage")
async def usage_stats(
    request: Request,
    client: str | None = None,
):
    """Return per-client usage statistics. No auth required (read-only operational data)."""
    db = request.app.state.db
    return get_usage(db, client_name=client)
