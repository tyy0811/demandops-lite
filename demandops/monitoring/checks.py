"""Input validation and monitoring checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    warnings: list[str]


def check_sparse_zone(
    zone_id: int,
    zone_mean_demands: dict[int, float],
    threshold_percentile: float = 10.0,
) -> list[str]:
    """Check if a zone is sparse (below P10 mean demand)."""
    warnings = []
    if zone_id in zone_mean_demands:
        values = sorted(zone_mean_demands.values())
        if values:
            idx = int(len(values) * threshold_percentile / 100)
            p10 = values[min(idx, len(values) - 1)]
            if zone_mean_demands[zone_id] < p10:
                warnings.append(
                    f"zone_id {zone_id} is a sparse zone "
                    f"(mean demand {zone_mean_demands[zone_id]:.1f} < P10={p10:.1f})"
                )
    return warnings


def check_extreme_prediction(
    predicted_count: float,
    zone_id: int,
    zone_max_demands: dict[int, float],
    threshold_multiplier: float = 5.0,
) -> list[str]:
    """Flag predictions unusually high for the zone."""
    warnings = []
    if zone_id in zone_max_demands:
        max_seen = zone_max_demands[zone_id]
        if max_seen > 0 and predicted_count > max_seen * threshold_multiplier:
            warnings.append(
                f"Prediction {predicted_count:.1f} exceeds "
                f"{threshold_multiplier}x max historical ({max_seen:.1f}) "
                f"for zone {zone_id}"
            )
    return warnings
