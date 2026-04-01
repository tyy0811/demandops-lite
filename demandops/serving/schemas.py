"""Pydantic v2 schemas for the serving API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    # No upper bound -- supports taxi zones (1-263) and TfL stations (higher IDs)
    zone_id: int = Field(ge=1)
    hour_ts: datetime


class PredictionMetadata(BaseModel):
    latency_ms: float
    request_id: str
    features_used: dict
    input_warnings: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    prediction_id: str
    zone_id: int
    zone_name: str
    hour_ts: datetime
    predicted_count: float = Field(ge=0.0)
    model_name: str
    model_version: str
    metadata: PredictionMetadata


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: str
    model_objective: str
    model_version: str
    history_loaded: bool
    supported_start: datetime | None = None
    supported_end: datetime | None = None
    n_supported_zones: int = 0
    history_rows: int = 0
    uptime_seconds: float


class ErrorDetail(BaseModel):
    detail: str
    supported_start: datetime | None = None
    supported_end: datetime | None = None
    n_supported_zones: int | None = None


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(min_length=1, max_length=10_000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    prediction_count: int
    latency_ms: float


class ActualSubmission(BaseModel):
    prediction_id: str | None = None
    zone_id: int | None = None
    hour_ts: datetime | None = None
    actual_value: float

    @model_validator(mode="after")
    def require_matching_key(self) -> "ActualSubmission":
        has_pid = self.prediction_id is not None
        has_zone_ts = self.zone_id is not None and self.hour_ts is not None
        if not has_pid and not has_zone_ts:
            raise ValueError(
                "Each actual must include either 'prediction_id' or both 'zone_id' and 'hour_ts'"
            )
        return self


class ActualsRequest(BaseModel):
    actuals: list[ActualSubmission] = Field(min_length=1)


class ActualsResponse(BaseModel):
    matched_count: int
    unmatched_count: int
    warnings: list[str] = Field(default_factory=list)
