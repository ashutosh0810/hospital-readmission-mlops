"""
train.py — Stage 3: train a baseline + advanced model, track with MLflow,
register and promote the best. Imported by Model_Development_and_Tracking.ipynb.

Run:  python -m src.train
"""
import config as cfg

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import data_prep as dp
from src.evaluate import compute_metrics

def build_logreg_pipeline(numeric_cols, categorical_cols):
    """
    Build the Stage 3.1 Logistic Regression baseline.

    The model is a FULL sklearn Pipeline containing:
        preprocessing -> Logistic Regression

    Class imbalance is handled using:
        class_weight='balanced'
    """

    preprocessor = dp.build_preprocessor(
        numeric_cols,
        categorical_cols
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=cfg.RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    return model



def build_xgb_pipeline(
    numeric_cols,
    categorical_cols,
    scale_pos_weight
):
    """
    Build the Stage 3.2 XGBoost advanced model.

    Full sklearn Pipeline:
        preprocessing -> XGBoost

    Class imbalance is handled using scale_pos_weight,
    calculated from the training target as:

        negative samples / positive samples
    """

    # Build a fresh unfitted preprocessor
    preprocessor = dp.build_preprocessor(
        numeric_cols,
        categorical_cols
    )

    # Build XGBoost classifier using the project
    # configuration plus imbalance handling
    classifier = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        **cfg.XGB_PARAMS
    )

    # Full leakage-safe sklearn pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    return model

def evaluate_model(model, X, y):
    """
    Generate positive-class probabilities and calculate
    imbalance-aware classification metrics.
    """

    probabilities = model.predict_proba(X)[:, 1]

    metrics = compute_metrics(
        y,
        probabilities
    )

    return probabilities, metrics

def train_and_log():
    """
    Complete Stage 3 training workflow.

    Workflow:
    1. Load fixed train / validation / test splits.
    2. Train Logistic Regression and XGBoost full Pipelines.
    3. Evaluate both on validation data.
    4. Track both experiments in MLflow.
    5. Select the best model using validation ROC-AUC.
    6. Evaluate ONLY the selected model on the test set.
    7. Register selected model and assign 'production' alias.
    8. Save artifacts required by Stage 4.

    Returns:
        (best_model_name, test_metrics)
    """

    # Local imports keep this function self-contained
    import json
    import joblib
    import mlflow
    import mlflow.sklearn

    from mlflow import MlflowClient
    from mlflow.models import infer_signature


    
    # 1. Get reproducible train / validation / test splits
    

    model_df = dp.get_model_frame()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = dp.get_splits(model_df)


    numeric_cols, categorical_cols = (
        dp.get_feature_lists(X_train)
    )


    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))


    
    # 2. Calculate class imbalance from TRAINING data only
    

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count / positive_count
    )


    print(
        "scale_pos_weight:",
        round(scale_pos_weight, 4)
    )


    
    # 3. Build both complete sklearn Pipelines
    

    logreg_model = build_logreg_pipeline(
        numeric_cols,
        categorical_cols
    )


    xgb_model = build_xgb_pipeline(
        numeric_cols,
        categorical_cols,
        scale_pos_weight
    )


    
    # 4. Configure MLflow
    

    mlflow.set_tracking_uri(
        cfg.MLFLOW_TRACKING_URI
    )

    mlflow.set_registry_uri(
        cfg.MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        cfg.MLFLOW_EXPERIMENT
    )


    # Stores results needed for final model selection
    results = {}


    
    # 5. Train + evaluate + log Logistic Regression
    

    print(
        "\nTraining Logistic Regression..."
    )

    logreg_model.fit(
        X_train,
        y_train
    )


    (
        logreg_val_prob,
        logreg_val_metrics
    ) = evaluate_model(
        logreg_model,
        X_val,
        y_val
    )


    with mlflow.start_run(
        run_name="logistic_regression_baseline"
    ) as run:

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": cfg.RANDOM_STATE
            }
        )


        mlflow.log_metrics(
            {
                f"val_{metric}": float(value)
                for metric, value
                in logreg_val_metrics.items()
            }
        )


        # Model signature + example remove MLflow
        # signature/input-example warnings.
        input_example = (
            X_train.head(5).copy()
        )

        signature = infer_signature(
            X_train.head(100),
            logreg_model.predict(
                X_train.head(100)
            )
        )


        model_info = mlflow.sklearn.log_model(
            sk_model=logreg_model,
            name="model",
            signature=signature,
            input_example=input_example
        )


        results["Logistic Regression"] = {
            "model": logreg_model,
            "validation_metrics":
                logreg_val_metrics,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri
        }


    print(
        "Logistic Regression validation metrics:",
        logreg_val_metrics
    )


    
    # 6. Train + evaluate + log XGBoost
    

    print(
        "\nTraining XGBoost..."
    )


    xgb_model.fit(
        X_train,
        y_train
    )


    (
        xgb_val_prob,
        xgb_val_metrics
    ) = evaluate_model(
        xgb_model,
        X_val,
        y_val
    )


    with mlflow.start_run(
        run_name="xgboost_advanced"
    ) as run:

        xgb_logged_params = {
            "model_type": "XGBoost",
            "scale_pos_weight":
                float(scale_pos_weight)
        }

        xgb_logged_params.update(
            cfg.XGB_PARAMS
        )


        mlflow.log_params(
            xgb_logged_params
        )


        mlflow.log_metrics(
            {
                f"val_{metric}": float(value)
                for metric, value
                in xgb_val_metrics.items()
            }
        )


        input_example = (
            X_train.head(5).copy()
        )

        signature = infer_signature(
            X_train.head(100),
            xgb_model.predict(
                X_train.head(100)
            )
        )


        model_info = mlflow.sklearn.log_model(
            sk_model=xgb_model,
            name="model",
            signature=signature,
            input_example=input_example
        )


        results["XGBoost"] = {
            "model": xgb_model,
            "validation_metrics":
                xgb_val_metrics,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri
        }


    print(
        "XGBoost validation metrics:",
        xgb_val_metrics
    )


    
    # 7. Select best model using VALIDATION ROC-AUC
    

    best_model_name = max(
        results,
        key=lambda name:
            results[name][
                "validation_metrics"
            ]["roc_auc"]
    )


    best_result = results[
        best_model_name
    ]

    best_model = best_result[
        "model"
    ]


    print(
        "\nSelected model:",
        best_model_name
    )

    print(
        "Selected validation ROC-AUC:",
        best_result[
            "validation_metrics"
        ]["roc_auc"]
    )


    
    # 8. Evaluate selected model ONCE on untouched test data
    

    (
        test_probabilities,
        test_metrics
    ) = evaluate_model(
        best_model,
        X_test,
        y_test
    )


    print(
        "\nFinal Test Metrics:"
    )

    for metric, value in test_metrics.items():
        print(
            f"{metric:<10}: {value:.4f}"
        )


    
    # 9. Register selected model
    

    registered_version = (
        mlflow.register_model(
            model_uri=
                best_result["model_uri"],
            name=
                cfg.REGISTERED_MODEL
        )
    )


    version_number = (
        registered_version.version
    )


    
    # 10. Assign production alias
    

    client = MlflowClient()


    client.set_registered_model_alias(
        name=cfg.REGISTERED_MODEL,
        alias="production",
        version=version_number
    )


    print(
        "\nRegistered model:",
        cfg.REGISTERED_MODEL
    )

    print(
        "Registered version:",
        version_number
    )

    print(
        "Production alias:",
        f"{cfg.REGISTERED_MODEL}@production"
    )


    
    # 11. Save Stage-4 operational artifacts
    

    cfg.ARTIFACT_DIR.mkdir(
        exist_ok=True
    )


    
    # best_model.pkl
    # Complete preprocessing + classifier Pipeline
    

    best_model_path = (
        cfg.ARTIFACT_DIR /
        "best_model.pkl"
    )

    joblib.dump(
        best_model,
        best_model_path
    )


    
    # metrics.json
    

    metrics_payload = {
        "best_model": best_model_name,
        "selection_metric": "roc_auc",
        "validation_metrics":
            best_result[
                "validation_metrics"
            ],
        "test_metrics": test_metrics,
        "registered_model":
            cfg.REGISTERED_MODEL,
        "registered_version":
            str(version_number),
        "production_alias":
            "production"
    }


    with open(
        cfg.ARTIFACT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics_payload,
            f,
            indent=4
        )


    
    # reference_sample.csv
    # Fixed training reference for Stage-4 drift monitoring
    

    reference_sample = (
        X_train.sample(
            n=min(
                5000,
                len(X_train)
            ),
            random_state=
                cfg.RANDOM_STATE
        )
        .copy()
    )


    reference_sample.to_csv(
        cfg.ARTIFACT_DIR /
        "reference_sample.csv",
        index=False
    )


    
    # input_columns.json
    # API uses this to align incoming request fields
    

    input_columns = list(
        X_train.columns
    )


    with open(
        cfg.ARTIFACT_DIR /
        "input_columns.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            input_columns,
            f,
            indent=4
        )


    
    # 12. Final verification
    

    print(
        "\nArtifacts created:"
    )

    for filename in [
        "best_model.pkl",
        "metrics.json",
        "reference_sample.csv",
        "input_columns.json"
    ]:

        path = (
            cfg.ARTIFACT_DIR /
            filename
        )

        print(
            f"{filename}:",
            path.exists()
        )


    print(
        "\nStage 3 training workflow completed successfully."
    )


    return (
        best_model_name,
        test_metrics
    )


if __name__ == "__main__":
    train_and_log()
