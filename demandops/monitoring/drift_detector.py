"""Data drift detection: PSI, KS test, correlation shift.

Accumulates feature vectors in a bounded deque. Computes drift metrics
on demand when /monitoring/drift is called — no background threads.
"""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path

import numpy as np
from scipy import stats

from demandops.features import FEATURE_COLUMNS

CONTINUOUS_FEATURES = [c for c in FEATURE_COLUMNS if c != "zone_id"]
CONTINUOUS_INDICES = [FEATURE_COLUMNS.index(c) for c in CONTINUOUS_FEATURES]

PSI_WARNING = 0.1
PSI_ALERT = 0.25
KS_ALPHA = 0.05
CORRELATION_WARNING = 0.1


def compute_psi(
    decile_boundaries: list[float],
    reference_bin_counts: np.ndarray,
    current_values: np.ndarray,
) -> float:
    """Population Stability Index between reference and current distributions."""
    current_bin_counts = np.histogram(current_values, bins=decile_boundaries)[0]
    eps = 1e-6
    n_bins = len(reference_bin_counts)
    ref_pct = (reference_bin_counts + eps) / (reference_bin_counts.sum() + eps * n_bins)
    cur_pct = (current_bin_counts + eps) / (current_bin_counts.sum() + eps * n_bins)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ks(
    reference_sample: np.ndarray, current_values: np.ndarray
) -> tuple[float, float]:
    """KS two-sample test. Returns (statistic, p_value)."""
    stat, p_value = stats.ks_2samp(reference_sample, current_values)
    return float(stat), float(p_value)


def compute_correlation_shift(
    reference_corr: np.ndarray, current_continuous: np.ndarray
) -> float:
    """Frobenius norm of correlation matrix difference, normalized by feature pairs."""
    current_corr = np.corrcoef(current_continuous, rowvar=False)
    diff = current_corr - reference_corr
    n = reference_corr.shape[0]
    n_pairs = n * (n - 1) / 2
    return float(np.linalg.norm(diff, "fro") / max(n_pairs, 1))


class DriftAccumulator:
    """Thread-safe bounded buffer for feature vectors."""

    def __init__(self, maxlen: int = 1000, min_samples: int = 100) -> None:
        self._lock = threading.Lock()
        self._buffer: collections.deque[list[float]] = collections.deque(maxlen=maxlen)
        self.min_samples = min_samples
        self.maxlen = maxlen

    def add(self, feature_vector: list[float]) -> None:
        with self._lock:
            self._buffer.append(feature_vector)

    def add_batch(self, feature_vectors: list[list[float]]) -> None:
        with self._lock:
            for v in feature_vectors:
                self._buffer.append(v)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_samples(self) -> np.ndarray | None:
        """Return accumulated samples as numpy array, or None if below minimum."""
        with self._lock:
            if len(self._buffer) < self.min_samples:
                return None
            return np.array(list(self._buffer))


class DriftDetector:
    """Computes drift metrics against training reference distributions."""

    def __init__(
        self,
        reference_path: Path,
        maxlen: int = 1000,
        min_samples: int = 100,
    ) -> None:
        self.accumulator = DriftAccumulator(maxlen=maxlen, min_samples=min_samples)
        ref = json.loads(reference_path.read_text())
        self._reference = ref
        self._ref_corr = np.array(ref["correlation_matrix"])

    def compute_drift(self) -> dict:
        """Compute drift metrics on accumulated samples. On-demand only."""
        samples = self.accumulator.get_samples()
        if samples is None:
            return {
                "status": "insufficient_samples",
                "collected": self.accumulator.count,
                "required": self.accumulator.min_samples,
            }

        result: dict = {
            "status": "ok",
            "collected": len(samples),
            "features": {},
        }

        for i, feature_name in enumerate(FEATURE_COLUMNS):
            feature_ref = self._reference["features"][feature_name]
            current_values = samples[:, i]

            psi = compute_psi(
                feature_ref["decile_boundaries"],
                np.array(feature_ref["bin_counts"]),
                current_values,
            )
            ks_stat, ks_pvalue = compute_ks(
                np.array(feature_ref["ks_subsample"]), current_values
            )

            if psi > PSI_ALERT or ks_pvalue < KS_ALPHA:
                verdict = "alert"
            elif psi > PSI_WARNING:
                verdict = "warning"
            else:
                verdict = "ok"

            result["features"][feature_name] = {
                "psi": round(psi, 6),
                "ks_statistic": round(ks_stat, 6),
                "ks_pvalue": round(ks_pvalue, 6),
                "verdict": verdict,
            }

        # Correlation shift on continuous features only
        continuous_samples = samples[:, CONTINUOUS_INDICES]
        corr_shift = compute_correlation_shift(self._ref_corr, continuous_samples)
        result["correlation_shift"] = round(corr_shift, 6)

        # Overall status
        verdicts = [f["verdict"] for f in result["features"].values()]
        if "alert" in verdicts or corr_shift > CORRELATION_WARNING:
            result["status"] = "alert"
        elif "warning" in verdicts:
            result["status"] = "warning"

        return result
