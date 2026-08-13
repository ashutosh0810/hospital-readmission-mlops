"""
retrain.py — Stage 4.5:
multi-signal retraining trigger + automated workflow.

Run:
    python -m src.retrain
"""

import json
from datetime import datetime, timezone

import config as cfg

from src.train import train_and_log


PSI_THRESHOLD = 0.20
DRIFT_SHARE_THRESHOLD = 0.30


def decide(summary):
    """
    Evaluate all retraining signals.

    Retrain when ANY of these conditions is true:

    1. prediction PSI > 0.20
    2. drifted feature share > 0.30
    3. Evidently dataset_drift is True

    Returns
    -------
    list[str]
        Human-readable reasons for retraining.
        Empty list means retraining is not required.
    """

    reasons = []

    prediction_psi = float(
        summary.get(
            "prediction_psi",
            0.0
        )
    )

    share_drifted = float(
        summary.get(
            "share_drifted",
            0.0
        )
    )

    dataset_drift = bool(
        summary.get(
            "dataset_drift",
            False
        )
    )

    if prediction_psi > PSI_THRESHOLD:

        reasons.append(
            f"Prediction PSI {prediction_psi:.4f} "
            f"> threshold {PSI_THRESHOLD:.2f}"
        )

    if share_drifted > DRIFT_SHARE_THRESHOLD:

        reasons.append(
            f"Drifted feature share "
            f"{share_drifted:.4f} "
            f"> threshold "
            f"{DRIFT_SHARE_THRESHOLD:.2f}"
        )

    if dataset_drift:

        reasons.append(
            "Evidently dataset_drift=True"
        )

    return reasons


def run_retraining_workflow():
    """
    Read drift-monitoring results and automatically retrain
    when one or more monitoring signals cross their thresholds.

    A decision record is always saved for governance.
    """

    print("=" * 60)
    print("STAGE 4.5 — RETRAINING WORKFLOW")
    print("=" * 60)

    summary_path = (
        cfg.ARTIFACT_DIR
        / "drift_summary.json"
    )

    decision_path = (
        cfg.ARTIFACT_DIR
        / "retraining_decision.json"
    )

    if not summary_path.exists():

        raise FileNotFoundError(
            "drift_summary.json not found. "
            "Run python -m src.monitoring first."
        )

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(file)

    print("\nMonitoring signals:")

    print(
        "Prediction PSI:",
        summary.get("prediction_psi")
    )

    print(
        "Drifted feature share:",
        summary.get("share_drifted")
    )

    print(
        "Dataset drift:",
        summary.get("dataset_drift")
    )

    reasons = decide(summary)

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # ---------------------------------------------------------
    # No trigger
    # ---------------------------------------------------------

    if not reasons:

        print("\nRetraining decision: NO RETRAINING")

        decision = {
            "timestamp_utc": timestamp,
            "retrain": False,
            "reasons": [],
            "monitoring_summary": summary,
            "model": None,
            "test_metrics": None,
        }

    # ---------------------------------------------------------
    # Triggered
    # ---------------------------------------------------------

    else:

        print("\nRetraining decision: RETRAIN")

        print("\nTrigger reason(s):")

        for reason in reasons:
            print("-", reason)

        print(
            "\nStarting automated retraining..."
        )

        best_model_name, test_metrics = (
            train_and_log()
        )

        decision = {
            "timestamp_utc": timestamp,
            "retrain": True,
            "reasons": reasons,
            "monitoring_summary": summary,
            "model": best_model_name,
            "test_metrics": {
                key: float(value)
                for key, value
                in test_metrics.items()
            },
        }

        print(
            "\nRetraining completed."
        )

        print(
            "Selected model:",
            best_model_name
        )

        print(
            "New test metrics:"
        )

        print(
            json.dumps(
                decision["test_metrics"],
                indent=4
            )
        )

    # ---------------------------------------------------------
    # Governance decision record
    # ---------------------------------------------------------

    with open(
        decision_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            decision,
            file,
            indent=4
        )

    print(
        "\nDecision record saved:",
        decision_path
    )

    print(
        "\nRetraining Decision Record:"
    )

    print(
        json.dumps(
            decision,
            indent=4
        )
    )

    return decision


if __name__ == "__main__":
    run_retraining_workflow()