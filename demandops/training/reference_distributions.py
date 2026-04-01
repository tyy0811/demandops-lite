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

    # Correlation matrix on continuous features only (zone_id excluded).
    # Exclude zero-variance columns to avoid NaN from np.corrcoef.
    cont_data = X_train[:, CONTINUOUS_INDICES]
    variances = np.var(cont_data, axis=0)
    valid_mask = variances > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) >= 2:
        corr = np.corrcoef(cont_data[:, valid_indices], rowvar=False)
    else:
        corr = np.eye(len(valid_indices)) if len(valid_indices) == 1 else np.array([[]])

    # Store as full-size matrix with NaN replaced by 0 for constant columns,
    # plus a mask so downstream consumers know which columns were valid.
    full_corr = np.zeros((len(CONTINUOUS_FEATURES), len(CONTINUOUS_FEATURES)))
    for i_out, i_in in enumerate(valid_indices):
        for j_out, j_in in enumerate(valid_indices):
            full_corr[i_in, j_in] = corr[i_out, j_out]

    ref["correlation_matrix"] = full_corr.tolist()
    ref["correlation_features"] = CONTINUOUS_FEATURES
    ref["correlation_valid_mask"] = valid_mask.tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ref, indent=2))
