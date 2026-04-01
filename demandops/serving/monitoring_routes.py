"""Monitoring API routes: /monitoring/drift, /monitoring/quality, /monitoring/actuals."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from demandops.security.auth import requires_auth
from demandops.serving.schemas import ActualsRequest, ActualsResponse

monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@monitoring_router.get("/drift")
async def drift_status(request: Request):
    """Return current drift status per feature. No auth required."""
    detector = request.app.state.drift_detector
    if detector is None:
        return {"status": "disabled", "reason": "reference distributions not loaded"}
    return detector.compute_drift()


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

    # Drift-quality correlation: if quality is degraded, include drift status
    if result.get("status") == "ok":
        monitoring_cfg = getattr(request.app.state, "monitoring_config", {})
        mae_threshold = monitoring_cfg.get("mae_threshold", 3.20)
        margin = monitoring_cfg.get("mae_alert_margin", 1.2)
        if result.get("mae", 0) > mae_threshold * margin:
            detector = request.app.state.drift_detector
            if detector is not None:
                drift = detector.compute_drift()
                result["drift_correlation"] = {
                    "drift_status": drift.get("status"),
                    "note": "Quality degradation detected alongside drift — may indicate retraining needed",
                }

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
