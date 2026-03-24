"""Tests for FeatureService."""

from __future__ import annotations

from datetime import datetime, timedelta

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
