"""Generate reference distribution artifact for drift detection.

Computes per-feature decile boundaries, KS subsamples, and the
correlation matrix on continuous features. Saved to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from demandops.features import FEATURE_COLUMNS

CONTINUOUS_FEATURES = [c for c in FEATURE_COLUMNS if c != "zone_id"]
CONTINUOUS_INDICES = [FEATURE_COLUMNS.index(c) for c in CONTINUOUS_FEATURES]


def generate_reference_distributions(
    X_train: np.ndarray,
    output_path: Path,
    ks_subsample_size: int = 5000,
    n_bins: int = 10,
    seed: int = 42,
) -> None:
    """Generate and save reference distributions from training data."""
    rng = np.random.RandomState(seed)
    ref: dict = {
        "features": {},
        "metadata": {
            "ks_subsample_size": ks_subsample_size,
            "n_bins": n_bins,
            "n_training_rows": len(X_train),
        },
    }

    for i, feature_name in enumerate(FEATURE_COLUMNS):
        col = X_train[:, i]

        quantiles = np.linspace(0, 100, n_bins + 1)
        boundaries = np.percentile(col, quantiles).tolist()
        bin_counts = np.histogram(col, bins=boundaries)[0].tolist()

        sample_size = min(ks_subsample_size, len(col))
        subsample_idx = rng.choice(len(col), sample_size, replace=False)
        subsample = col[subsample_idx].tolist()

        ref["features"][feature_name] = {
            "decile_boundaries": boundaries,
            "bin_counts": bin_counts,
            "ks_subsample": subsample,
        }

    # Correlation matrix on continuous features only (zone_id excluded)
    cont_data = X_train[:, CONTINUOUS_INDICES]
    ref["correlation_matrix"] = np.corrcoef(cont_data, rowvar=False).tolist()
    ref["correlation_features"] = CONTINUOUS_FEATURES

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ref, indent=2))
