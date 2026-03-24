"""Tests for FeatureService."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from demandops.serving.feature_service import FeatureService


class TestFeatureService:

    def test_valid_zone_and_timestamp(self, feature_service: FeatureService) -> None:
        result = feature_service.get_features(1, datetime(2024, 2, 1, 12, 0))
        assert result.supported
        assert result.features is not None
        assert len(result.features) == 9

    def test_zone_not_in_universe(self, feature_service: FeatureService) -> None:
        result = feature_service.get_features(999, datetime(2024, 2, 1, 12, 0))
        assert not result.supported
        assert result.features is None
        assert any("zone universe" in w for w in result.warnings)

    def test_zone_in_range_but_not_in_universe(
        self, feature_service: FeatureService
    ) -> None:
        """Zone 100 is in 1-263 but not in test universe [1,2,3]."""
        result = feature_service.get_features(100, datetime(2024, 2, 1, 12, 0))
        assert not result.supported

    def test_timestamp_before_supported_start(
        self, feature_service: FeatureService
    ) -> None:
        result = feature_service.get_features(1, datetime(2023, 12, 15, 12, 0))
        assert not result.supported

    def test_timestamp_at_supported_end_exclusive(
        self, feature_service: FeatureService
    ) -> None:
        result = feature_service.get_features(1, feature_service.supported_end)
        assert not result.supported

    def test_supported_start(self, feature_service: FeatureService) -> None:
        assert feature_service.supported_start == datetime(2024, 1, 1)

    def test_supported_end(self, feature_service: FeatureService) -> None:
        assert feature_service.supported_end == datetime(2024, 3, 1)

    def test_n_supported_zones(self, feature_service: FeatureService) -> None:
        assert feature_service.n_supported_zones == 3

    def test_feature_order_matches_schema(
        self, feature_service: FeatureService
    ) -> None:
        from demandops.features import FEATURE_COLUMNS
        result = feature_service.get_features(1, datetime(2024, 2, 1, 12, 0))
        assert result.supported
        assert list(result.features.keys()) == FEATURE_COLUMNS

    def test_weekday_matches_python_convention(
        self, feature_service: FeatureService
    ) -> None:
        """FeatureService uses datetime.weekday() → 0=Mon.
        Monday 2024-01-01 should have day_of_week=0."""
        result = feature_service.get_features(1, datetime(2024, 1, 1, 12, 0))
        assert result.supported
        assert result.features["day_of_week"] == 0  # Monday

    def test_last_supported_hour(self, feature_service: FeatureService) -> None:
        last_hour = feature_service.supported_end - timedelta(hours=1)
        result = feature_service.get_features(1, last_hour)
        assert result.supported

    def test_timezone_aware_converts_to_utc(
        self, feature_service: FeatureService
    ) -> None:
        """Timezone-aware timestamps must be converted to UTC, not just stripped.

        2024-02-01T14:00:00+02:00 == 2024-02-01T12:00:00 UTC.
        Both should produce the same features.
        """
        naive_utc = datetime(2024, 2, 1, 12, 0, 0)
        tz_plus2 = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone(timedelta(hours=2)))

        result_naive = feature_service.get_features(1, naive_utc)
        result_tz = feature_service.get_features(1, tz_plus2)

        assert result_naive.supported
        assert result_tz.supported
        assert result_naive.features == result_tz.features

    def test_timezone_utc_equivalent_to_naive(
        self, feature_service: FeatureService
    ) -> None:
        """UTC-tagged timestamp produces same result as naive."""
        naive = datetime(2024, 2, 1, 12, 0, 0)
        utc_aware = datetime(2024, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

        result_naive = feature_service.get_features(1, naive)
        result_utc = feature_service.get_features(1, utc_aware)

        assert result_naive.features == result_utc.features

    def test_mismatched_schema_raises_at_init(
        self,
        history_parquet_path: Path,
        zone_universe_path: Path,
        split_config: dict,
        tmp_path: Path,
    ) -> None:
        """FeatureService must reject a schema artifact with wrong column order."""
        bad_schema = {
            "columns": ["zone_id", "hour_of_day"],  # wrong order/count
            "target": "trip_count",
        }
        bad_path = tmp_path / "bad_schema.json"
        bad_path.write_text(json.dumps(bad_schema))

        with pytest.raises(ValueError, match="does not match"):
            FeatureService(
                history_path=history_parquet_path,
                schema_path=bad_path,
                zone_universe_path=zone_universe_path,
                config=split_config,
            )
