"""Tests for data drift detection."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from demandops.features import FEATURE_COLUMNS


@pytest.fixture
def reference_distributions(tmp_path: Path) -> Path:
    """Build a synthetic reference distribution artifact."""
    rng = np.random.RandomState(42)
    n_samples = 5000

    ref = {"features": {}, "metadata": {"ks_subsample_size": n_samples, "n_bins": 10}}

    for feature_name in FEATURE_COLUMNS:
        if feature_name == "zone_id":
            values = rng.choice([1, 2, 3, 4, 5], size=n_samples).astype(float)
        elif feature_name == "hour_of_day":
            values = rng.choice(range(24), size=n_samples).astype(float)
        elif feature_name == "day_of_week":
            values = rng.choice(range(7), size=n_samples).astype(float)
        elif feature_name == "is_weekend":
            values = rng.choice([0, 1], size=n_samples, p=[5 / 7, 2 / 7]).astype(float)
        elif feature_name == "month":
            values = rng.choice(range(1, 13), size=n_samples).astype(float)
        else:
            values = rng.exponential(5, size=n_samples)

        quantiles = np.linspace(0, 100, 11)
        boundaries = np.percentile(values, quantiles).tolist()
        bin_counts = np.histogram(values, bins=boundaries)[0].tolist()

        ref["features"][feature_name] = {
            "decile_boundaries": boundaries,
            "bin_counts": bin_counts,
            "ks_subsample": values.tolist(),
        }

    # Correlation matrix on continuous features only
    cont_features = [c for c in FEATURE_COLUMNS if c != "zone_id"]
    cont_indices = [FEATURE_COLUMNS.index(c) for c in cont_features]
    full_matrix = np.column_stack([
        np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS
    ])
    cont_matrix = full_matrix[:, cont_indices]
    ref["correlation_matrix"] = np.corrcoef(cont_matrix, rowvar=False).tolist()
    ref["correlation_features"] = cont_features

    path = tmp_path / "reference_distributions.json"
    path.write_text(json.dumps(ref))
    return path


class TestPSI:
    def test_psi_triggers_on_shifted_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_psi

        ref = json.loads(reference_distributions.read_text())
        feature_ref = ref["features"]["hour_of_day"]

        # Shift: all values in 0-6 (night hours only)
        shifted = np.random.RandomState(99).choice(range(0, 7), size=500).astype(float)
        psi = compute_psi(
            feature_ref["decile_boundaries"],
            np.array(feature_ref["bin_counts"]),
            shifted,
        )
        assert psi > 0.25, f"Expected PSI > 0.25 for shifted data, got {psi}"

    def test_psi_low_on_same_distribution(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_psi

        ref = json.loads(reference_distributions.read_text())
        feature_ref = ref["features"]["hour_of_day"]

        # Same distribution as reference
        same = np.random.RandomState(99).choice(range(24), size=500).astype(float)
        psi = compute_psi(
            feature_ref["decile_boundaries"],
            np.array(feature_ref["bin_counts"]),
            same,
        )
        assert psi < 0.1, f"Expected PSI < 0.1 for same distribution, got {psi}"


class TestKS:
    def test_ks_triggers_on_shifted_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_ks

        ref = json.loads(reference_distributions.read_text())
        ref_sample = np.array(ref["features"]["lag_1h"]["ks_subsample"])

        # Different distribution
        shifted = np.random.RandomState(99).normal(50, 1, size=500)
        _, p_value = compute_ks(ref_sample, shifted)
        assert p_value < 0.05

    def test_ks_no_alert_on_same_distribution(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_ks

        ref = json.loads(reference_distributions.read_text())
        ref_sample = np.array(ref["features"]["lag_1h"]["ks_subsample"])

        # Subsample from same reference
        same = np.random.RandomState(99).choice(ref_sample, size=500, replace=True)
        _, p_value = compute_ks(ref_sample, same)
        assert p_value > 0.05


class TestCorrelationShift:
    def test_detects_altered_correlation(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_correlation_shift

        ref = json.loads(reference_distributions.read_text())
        ref_corr = np.array(ref["correlation_matrix"])

        # Build strongly correlated samples that differ from the reference
        rng = np.random.RandomState(42)
        n = 500
        n_cont = ref_corr.shape[0]
        base = rng.randn(n, 1)
        # All features strongly correlated — creates a near-1 correlation matrix
        samples = np.hstack([base + rng.randn(n, 1) * 0.1 for _ in range(n_cont)])

        shift, n_excluded = compute_correlation_shift(ref_corr, samples)
        assert shift > 0.01, f"Expected correlation shift > 0.01, got {shift}"
        assert n_excluded == 0

    def test_low_shift_on_matching_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import compute_correlation_shift

        ref = json.loads(reference_distributions.read_text())
        ref_corr = np.array(ref["correlation_matrix"])
        cont_features = [c for c in FEATURE_COLUMNS if c != "zone_id"]
        cont_indices = [FEATURE_COLUMNS.index(c) for c in cont_features]

        # Use the reference data itself
        full_matrix = np.column_stack([
            np.array(ref["features"][f]["ks_subsample"]) for f in FEATURE_COLUMNS
        ])
        cont_samples = full_matrix[:, cont_indices]

        shift, n_excluded = compute_correlation_shift(ref_corr, cont_samples)
        assert shift < 0.05, f"Expected low correlation shift, got {shift}"
        assert n_excluded == 0

    def test_zero_variance_columns_excluded(self, reference_distributions) -> None:
        """Constant columns (e.g., all same hour) should not produce NaN."""
        from demandops.monitoring.drift_detector import compute_correlation_shift

        ref = json.loads(reference_distributions.read_text())
        ref_corr = np.array(ref["correlation_matrix"])
        n_cont = ref_corr.shape[0]

        # Create data where 3 columns are constant (hour, day_of_week, is_weekend)
        rng = np.random.RandomState(42)
        n = 100
        samples = rng.randn(n, n_cont)
        samples[:, 0] = 12.0   # hour_of_day constant
        samples[:, 1] = 3.0    # day_of_week constant
        samples[:, 2] = 0.0    # is_weekend constant

        shift, n_excluded = compute_correlation_shift(ref_corr, samples)
        assert not np.isnan(shift), "correlation_shift should not be NaN"
        assert n_excluded == 3

    def test_all_constant_returns_zero(self, reference_distributions) -> None:
        """If all columns are constant, shift should be 0, not NaN."""
        from demandops.monitoring.drift_detector import compute_correlation_shift

        ref = json.loads(reference_distributions.read_text())
        ref_corr = np.array(ref["correlation_matrix"])
        n_cont = ref_corr.shape[0]

        samples = np.ones((100, n_cont))  # All constant
        shift, n_excluded = compute_correlation_shift(ref_corr, samples)
        assert shift == 0.0
        assert n_excluded == n_cont

    def test_nan_in_reference_excluded(self) -> None:
        """Reference matrix with NaN (constant training column) doesn't produce NaN."""
        from demandops.monitoring.drift_detector import compute_correlation_shift

        # Simulate a reference matrix where column 2 was constant during training
        n = 8
        ref_corr = np.eye(n)
        ref_corr[2, :] = np.nan
        ref_corr[:, 2] = np.nan

        rng = np.random.RandomState(42)
        samples = rng.randn(100, n)

        shift, n_excluded = compute_correlation_shift(ref_corr, samples)
        assert not np.isnan(shift), "shift should not be NaN"
        assert n_excluded >= 1  # Column 2 should be excluded


class TestDriftAccumulator:
    def test_insufficient_samples_returns_none(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=100)
        for _ in range(50):
            acc.add([1.0] * 9)
        assert acc.get_samples() is None
        assert acc.count == 50

    def test_returns_samples_above_threshold(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=100)
        for _ in range(100):
            acc.add([1.0] * 9)
        samples = acc.get_samples()
        assert samples is not None
        assert samples.shape == (100, 9)

    def test_deque_boundary_evicts_oldest(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=10)
        # Push 1200 samples with value = index
        for i in range(1200):
            acc.add([float(i)] * 9)

        assert acc.count == 1000
        samples = acc.get_samples()
        assert samples is not None
        # Oldest 200 should be evicted; first sample should be index 200
        assert samples[0, 0] == 200.0
        assert samples[-1, 0] == 1199.0

    def test_concurrent_writes_no_corruption(self) -> None:
        from demandops.monitoring.drift_detector import DriftAccumulator

        acc = DriftAccumulator(maxlen=1000, min_samples=10)

        def writer(thread_id: int) -> None:
            for i in range(100):
                acc.add([float(thread_id * 1000 + i)] * 9)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert acc.count == 1000

        samples = acc.get_samples()
        assert samples is not None
        # Every row should have 9 identical values (no partial vectors)
        for row in samples:
            assert len(set(row)) == 1, f"Corrupted vector: {row}"


class TestDriftDetector:
    def test_insufficient_samples_response(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import DriftDetector

        detector = DriftDetector(reference_distributions, min_samples=100)
        for _ in range(50):
            detector.accumulator.add([1.0] * 9)

        result = detector.compute_drift()
        assert result["status"] == "insufficient_samples"
        assert result["collected"] == 50
        assert result["required"] == 100

    def test_no_drift_on_reference_data(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import DriftDetector

        ref = json.loads(reference_distributions.read_text())
        detector = DriftDetector(reference_distributions, min_samples=50)

        # Feed reference data back
        rng = np.random.RandomState(42)
        for _ in range(200):
            vector = []
            for feature_name in FEATURE_COLUMNS:
                subsample = ref["features"][feature_name]["ks_subsample"]
                vector.append(rng.choice(subsample))
            detector.accumulator.add(vector)

        result = detector.compute_drift()
        assert result["status"] == "ok"
        for feature_name, metrics in result["features"].items():
            assert metrics["verdict"] == "ok", (
                f"False alarm on {feature_name}: PSI={metrics['psi']}, "
                f"KS p={metrics['ks_pvalue']}"
            )

    def test_detects_shifted_feature(self, reference_distributions) -> None:
        from demandops.monitoring.drift_detector import DriftDetector

        ref = json.loads(reference_distributions.read_text())
        detector = DriftDetector(reference_distributions, min_samples=50)

        rng = np.random.RandomState(42)
        hour_idx = FEATURE_COLUMNS.index("hour_of_day")

        for _ in range(200):
            vector = []
            for j, feature_name in enumerate(FEATURE_COLUMNS):
                if j == hour_idx:
                    vector.append(float(rng.choice(range(0, 3))))  # Extreme shift
                else:
                    subsample = ref["features"][feature_name]["ks_subsample"]
                    vector.append(rng.choice(subsample))
            detector.accumulator.add(vector)

        result = detector.compute_drift()
        assert result["features"]["hour_of_day"]["verdict"] == "alert"
        assert result["status"] in ("alert", "warning")
