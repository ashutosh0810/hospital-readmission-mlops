"""
app.py — Stage 4.1: FastAPI inference service.

Endpoints:
    GET  /health
    POST /predict

Run locally:
    uvicorn app:app --reload
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

ARTIFACTS = (
    Path(__file__).resolve().parent
    / "artifacts"
)

THRESHOLD = 0.5


app = FastAPI(
    title="Hospital Readmission Predictor",
    version="1.0"
)


# ============================================================
# Request schema
# ============================================================

class PredictRequest(BaseModel):

    features: dict = Field(
        ...,
        description=(
            "Encounter features "
            "(column -> value)"
        )
    )


# ============================================================
# Lazy-loaded artifacts
# ============================================================

_MODEL = None
_INPUT_COLUMNS = None


def _load_artifacts():
    """
    Lazily load the trained model Pipeline and
    expected input columns.

    Artifacts are loaded only when the API is first used.
    """

    global _MODEL
    global _INPUT_COLUMNS


    # --------------------------------------------------------
    # Load trained Pipeline
    # --------------------------------------------------------

    if _MODEL is None:

        model_path = (
            ARTIFACTS
            / "best_model.pkl"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model artifact not found: "
                f"{model_path}"
            )

        _MODEL = joblib.load(
            model_path
        )


    # --------------------------------------------------------
    # Load expected input columns
    # --------------------------------------------------------

    if _INPUT_COLUMNS is None:

        columns_path = (
            ARTIFACTS
            / "input_columns.json"
        )

        if not columns_path.exists():

            raise FileNotFoundError(
                f"Input-column artifact "
                f"not found: {columns_path}"
            )

        with open(
            columns_path,
            "r",
            encoding="utf-8"
        ) as f:

            _INPUT_COLUMNS = json.load(f)


    return (
        _MODEL,
        _INPUT_COLUMNS
    )


# ============================================================
# Prediction logging
# ============================================================

def _log_prediction(
    probability,
    prediction
):
    """
    Append prediction metadata to predictions.log.

    This provides Stage 4 governance/audit evidence.
    """

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )


    log_record = {

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "readmission_probability":
            float(probability),

        "readmitted_30d":
            int(prediction),

        "threshold":
            float(THRESHOLD)
    }


    log_path = (
        ARTIFACTS
        / "predictions.log"
    )


    with open(
        log_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                log_record
            )
            + "\n"
        )


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
def health():
    """
    Check API and model availability.
    """

    try:

        model, input_columns = (
            _load_artifacts()
        )

        return {

            "status": "ok",

            "model_loaded":
                model is not None,

            "input_columns_loaded":
                bool(input_columns)
        }


    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts could "
                f"not be loaded: {exc}"
            )
        )


# ============================================================
# POST /predict
# ============================================================

@app.post("/predict")
def predict(
    request: PredictRequest
):
    """
    Generate 30-day readmission probability and label.

    Missing fields are converted to NaN so that the
    preprocessing Pipeline can perform imputation.
    """


    # --------------------------------------------------------
    # Request validation
    # --------------------------------------------------------

    if not request.features:

        raise HTTPException(
            status_code=422,
            detail=(
                "features must not be empty"
            )
        )


    try:

        model, input_columns = (
            _load_artifacts()
        )


        # ----------------------------------------------------
        # Align request with training feature columns
        # ----------------------------------------------------

        aligned_features = {}


        for column in input_columns:

            value = request.features.get(
                column,
                np.nan
            )

            # JSON null becomes Python None.
            # Convert None to NaN so sklearn imputers
            # can handle the missing value.
            if value is None:

                value = np.nan


            aligned_features[
                column
            ] = value


        input_df = pd.DataFrame(
            [aligned_features],
            columns=input_columns
        )


        # ----------------------------------------------------
        # Generate probability
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(
                input_df
            )[0, 1]
        )


        # ----------------------------------------------------
        # Apply classification threshold
        # ----------------------------------------------------

        prediction = int(
            probability
            >= THRESHOLD
        )


        # ----------------------------------------------------
        # Governance logging
        # ----------------------------------------------------

        _log_prediction(
            probability,
            prediction
        )


        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        return {

            "readmission_probability":
                probability,

            "readmitted_30d":
                prediction,

            "threshold":
                THRESHOLD
        }


    except HTTPException:

        raise


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{exc}"
            )
        )