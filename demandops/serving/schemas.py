"""Pydantic v2 schemas for the serving API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    zone_id: int = Field(ge=1, le=263)
    hour_ts: datetime


class PredictionMetadata(BaseModel):
    latency_ms: float
    request_id: str
    features_used: dict
    input_warnings: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
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
    zones_supported: list[int] = Field(default_factory=list)
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
