"""Prometheus metric definitions using prometheus-client."""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "demandops_requests_total",
    "Total HTTP requests",
    ["endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "demandops_predictions_total",
    "Total successful predictions",
)

REJECTION_COUNT = Counter(
    "demandops_rejections_total",
    "Total rejected requests (unsupported zone or timestamp)",
    ["reason"],
)

ERROR_COUNT = Counter(
    "demandops_errors_total",
    "Total internal errors",
)

REQUEST_LATENCY = Histogram(
    "demandops_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_VALUE = Histogram(
    "demandops_prediction_value",
    "Distribution of predicted trip counts",
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

MODEL_LOADED = Gauge(
    "demandops_model_loaded",
    "Whether the model is loaded (1=yes, 0=no)",
)

HISTORY_LOADED = Gauge(
    "demandops_history_loaded",
    "Whether the history table is loaded (1=yes, 0=no)",
)

# Drift monitoring metrics
DRIFT_PSI = Gauge(
    "demandops_drift_psi",
    "Current PSI value per feature",
    ["feature"],
)

DRIFT_KS_PVALUE = Gauge(
    "demandops_drift_ks_pvalue",
    "Current KS test p-value per feature",
    ["feature"],
)

DRIFT_ALERT = Gauge(
    "demandops_drift_alert",
    "Whether drift is detected per feature (1=alert, 0.5=warning, 0=ok)",
    ["feature"],
)

DRIFT_CORRELATION_SHIFT = Gauge(
    "demandops_drift_correlation_shift",
    "Frobenius norm of correlation matrix difference",
)

# Quality monitoring metrics
QUALITY_MAE = Gauge(
    "demandops_quality_mae",
    "Rolling MAE over matched prediction-actual pairs",
)

QUALITY_RMSE = Gauge(
    "demandops_quality_rmse",
    "Rolling RMSE over matched prediction-actual pairs",
)

QUALITY_SMAPE = Gauge(
    "demandops_quality_smape",
    "Rolling sMAPE over matched prediction-actual pairs",
)

QUALITY_ALERT = Gauge(
    "demandops_quality_alert",
    "Whether quality degradation is detected (1=yes, 0=no)",
)
