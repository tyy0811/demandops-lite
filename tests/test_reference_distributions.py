"""Tests for reference distribution artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from demandops.features import FEATURE_COLUMNS


@pytest.fixture
def training_features() -> np.ndarray:
    """Synthetic training feature matrix (1000 rows x 9 features)."""
    rng = np.random.RandomState(42)
    n = 1000
    return np.column_stack([
        rng.choice(range(24), size=n),        # hour_of_day
        rng.choice(range(7), size=n),          # day_of_week
        rng.choice([0, 1], size=n),            # is_weekend
        rng.choice(range(1, 13), size=n),      # month
        rng.choice([1, 2, 3], size=n),         # zone_id
        rng.exponential(5, size=n),            # lag_1h
        rng.exponential(5, size=n),            # lag_24h
        rng.exponential(5, size=n),            # lag_168h
        rng.exponential(5, size=n),            # rolling_mean_24h
    ]).astype(float)


class TestGenerateReferenceDistributions:
    def test_creates_artifact_file(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        assert output_path.exists()

    def test_contains_all_features(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        ref = json.loads(output_path.read_text())
        for feature_name in FEATURE_COLUMNS:
            assert feature_name in ref["features"]

    def test_decile_boundaries_have_11_values(
        self, tmp_path: Path, training_features
    ) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        ref = json.loads(output_path.read_text())
        for feature_name in FEATURE_COLUMNS:
            assert len(ref["features"][feature_name]["decile_boundaries"]) == 11

    def test_ks_subsample_capped(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(
            training_features, output_path, ks_subsample_size=500
        )
        ref = json.loads(output_path.read_text())
        for feature_name in FEATURE_COLUMNS:
            assert len(ref["features"][feature_name]["ks_subsample"]) == 500

    def test_correlation_matrix_8x8(self, tmp_path: Path, training_features) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(training_features, output_path)
        ref = json.loads(output_path.read_text())
        corr = np.array(ref["correlation_matrix"])
        assert corr.shape == (8, 8)

    def test_metadata_records_subsample_size(
        self, tmp_path: Path, training_features
    ) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(
            training_features, output_path, ks_subsample_size=500
        )
        ref = json.loads(output_path.read_text())
        assert ref["metadata"]["ks_subsample_size"] == 500
        assert ref["metadata"]["n_training_rows"] == 1000

    def test_constant_column_produces_no_nan(
        self, tmp_path: Path, training_features
    ) -> None:
        from demandops.training.reference_distributions import (
            generate_reference_distributions,
        )

        # Make "month" constant (index 3) — simulates single-month training split
        features_with_constant = training_features.copy()
        features_with_constant[:, 3] = 1.0  # month = January

        output_path = tmp_path / "ref.json"
        generate_reference_distributions(features_with_constant, output_path)
        ref = json.loads(output_path.read_text())
        corr = np.array(ref["correlation_matrix"])
        assert not np.any(np.isnan(corr)), "Correlation matrix should not contain NaN"
        # The constant column should have zeros in its row/column
        assert corr[3, 3] == 0.0
        # Valid mask should mark the constant column as invalid
        assert ref["correlation_valid_mask"][3] is False
