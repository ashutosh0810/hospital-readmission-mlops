"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.
Compares a production 'current' batch against the saved training reference.

Run:  python -m src.monitoring
"""
import config as cfg


"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.

Compares a production 'current' batch against the saved
training reference.

Run:
    python -m src.monitoring
"""

import json

import joblib
import numpy as np
import pandas as pd

import config as cfg

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


def calculate_psi(reference_scores, current_scores, bins=10):
    """
    Calculate Population Stability Index (PSI).

    Reference-score quantiles define the bins.
    """

    reference_scores = np.asarray(reference_scores, dtype=float)
    current_scores = np.asarray(current_scores, dtype=float)

    # Quantile-based breakpoints from reference distribution
    quantiles = np.linspace(0, 1, bins + 1)

    breakpoints = np.unique(
        np.quantile(reference_scores, quantiles)
    )

    # Defensive fallback in case too few unique breakpoints exist
    if len(breakpoints) < 3:
        minimum = min(
            reference_scores.min(),
            current_scores.min()
        )

        maximum = max(
            reference_scores.max(),
            current_scores.max()
        )

        breakpoints = np.linspace(
            minimum,
            maximum,
            bins + 1
        )

    # Ensure all future values fall into a bin
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_count, _ = np.histogram(
        reference_scores,
        bins=breakpoints
    )

    current_count, _ = np.histogram(
        current_scores,
        bins=breakpoints
    )

    reference_pct = (
        reference_count / reference_count.sum()
    )

    current_pct = (
        current_count / current_count.sum()
    )

    # Avoid log(0)
    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def run_monitoring():

    print("=" * 60)
    print("STAGE 4.4 — MONITORING & DRIFT DETECTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load reference/current datasets
    # ---------------------------------------------------------

    reference_path = (
        cfg.ARTIFACT_DIR / "reference_sample.csv"
    )

    current_path = (
        cfg.ARTIFACT_DIR / "current_batch.csv"
    )

    model_path = (
        cfg.ARTIFACT_DIR / "best_model.pkl"
    )

    input_columns_path = (
        cfg.ARTIFACT_DIR / "input_columns.json"
    )

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    with open(
        input_columns_path,
        "r",
        encoding="utf-8"
    ) as file:

        input_columns = json.load(file)

    # Ensure identical columns/order
    reference = reference[input_columns].copy()
    current = current[input_columns].copy()

    print(
        "\nReference shape:",
        reference.shape
    )

    print(
        "Current shape:",
        current.shape
    )

    print(
        "Number of monitored features:",
        len(input_columns)
    )

    # ---------------------------------------------------------
    # 2. FEATURE DRIFT — Evidently
    # ---------------------------------------------------------

    print("\nRunning Evidently feature drift...")

    drift_report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    drift_report.run(
        reference_data=reference,
        current_data=current
    )

    drift_report_path = (
        cfg.ARTIFACT_DIR / "drift_report.html"
    )

    drift_report.save_html(
        str(drift_report_path)
    )

    drift_result = drift_report.as_dict()

    # DataDriftPreset puts DatasetDriftMetric first
    dataset_drift_result = (
        drift_result["metrics"][0]["result"]
    )

    number_of_drifted_columns = int(
        dataset_drift_result[
            "number_of_drifted_columns"
        ]
    )

    number_of_columns = int(
        dataset_drift_result[
            "number_of_columns"
        ]
    )

    share_drifted = float(
        dataset_drift_result[
            "share_of_drifted_columns"
        ]
    )

    dataset_drift = bool(
        dataset_drift_result[
            "dataset_drift"
        ]
    )

    print(
        "\nFeature drift:"
    )

    print(
        f"Drifted columns: "
        f"{number_of_drifted_columns}/"
        f"{number_of_columns}"
    )

    print(
        f"Share drifted: "
        f"{share_drifted:.4f}"
    )

    print(
        f"Dataset drift: "
        f"{dataset_drift}"
    )

    # ---------------------------------------------------------
    # 3. PREDICTION DRIFT — PSI
    # ---------------------------------------------------------

    print(
        "\nCalculating prediction-score PSI..."
    )

    model = joblib.load(model_path)

    reference_scores = model.predict_proba(
        reference
    )[:, 1]

    current_scores = model.predict_proba(
        current
    )[:, 1]

    prediction_psi = calculate_psi(
        reference_scores,
        current_scores
    )

    print(
        f"Prediction PSI: "
        f"{prediction_psi:.4f}"
    )

    # ---------------------------------------------------------
    # 4. Interpretation
    # ---------------------------------------------------------

    if prediction_psi > 0.20:

        psi_interpretation = (
            "Significant prediction drift detected "
            "(PSI > 0.20)."
        )

    elif prediction_psi > 0.10:

        psi_interpretation = (
            "Moderate prediction drift detected "
            "(0.10 < PSI <= 0.20)."
        )

    else:

        psi_interpretation = (
            "Prediction distribution is stable "
            "(PSI <= 0.10)."
        )

    print(
        "Interpretation:",
        psi_interpretation
    )

    # ---------------------------------------------------------
    # 5. Save machine-readable summary
    # ---------------------------------------------------------

    summary = {
        "number_of_drifted_columns":
            number_of_drifted_columns,

        "number_of_columns":
            number_of_columns,

        "share_drifted":
            share_drifted,

        "dataset_drift":
            dataset_drift,

        "prediction_psi":
            prediction_psi,

        "prediction_psi_interpretation":
            psi_interpretation,
    }

    summary_path = (
        cfg.ARTIFACT_DIR / "drift_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        "\nSaved:",
        drift_report_path
    )

    print(
        "Saved:",
        summary_path
    )

    print(
        "\nMonitoring Summary:"
    )

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    return summary


if __name__ == "__main__":
    run_monitoring()


