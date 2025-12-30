"""
src/train.py

Main training script:
 - Parent MLflow run
 - Nested runs per model
 - Safe MLflow logging (no registry, no deprecated args)
"""

import os
import tempfile
import warnings
from math import prod

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pipeline import pipelines, param_spaces, search_type
from data import load_heart_data
from utils_plot import save_cm, save_roc

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# -------------------------
# Experiment configuration
# -------------------------
EXPERIMENT_NAME = "HeartDisease_Models"
mlflow.set_experiment(EXPERIMENT_NAME)

N_JOBS = -1
CV_FOLDS = 5
RANDOM_STATE = 42


def total_param_combinations(param_grid):
    sizes = [len(v) for v in param_grid.values()]
    return prod(sizes) if sizes else 1


def main():
    print("🚀 Training started")

    with mlflow.start_run(run_name="Main_Training_Run"):

        # Load data + EDA
        X, y, _ = load_heart_data(run_eda=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("cv_folds", CV_FOLDS)

        # -------------------------
        # Iterate over models
        # -------------------------
        for name, pipe in pipelines.items():
            print(f"\n🔹 Starting hyperparameter search for {name}")

            Searcher = search_type.get(name)
            params = param_spaces.get(name, {})

            if Searcher is None:
                print(f"⚠ No searcher defined for {name}, skipping.")
                continue

            if Searcher == GridSearchCV:
                search = GridSearchCV(
                    pipe,
                    params,
                    cv=CV_FOLDS,
                    n_jobs=N_JOBS,
                    scoring="roc_auc",
                )
            else:
                max_combos = total_param_combinations(params)
                n_iter = min(25, max_combos)

                search = RandomizedSearchCV(
                    pipe,
                    params,
                    n_iter=n_iter,
                    cv=CV_FOLDS,
                    n_jobs=N_JOBS,
                    scoring="roc_auc",
                    random_state=RANDOM_STATE,
                )

            # -------------------------
            # Nested MLflow run
            # -------------------------
            with mlflow.start_run(run_name=name, nested=True):

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                y_pred = best_model.predict(X_test)

                try:
                    y_score = best_model.predict_proba(X_test)[:, 1]
                except Exception:
                    try:
                        y_score = best_model.decision_function(X_test)
                    except Exception:
                        y_score = y_pred

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_score)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc)

                for k, v in search.best_params_.items():
                    mlflow.log_param(k, v)

                tmpdir = tempfile.mkdtemp()

                cm_path = os.path.join(tmpdir, f"{name}_cm.png")
                save_cm(y_test, y_pred, cm_path)
                mlflow.log_artifact(cm_path, artifact_path="artifacts")

                roc_path = os.path.join(tmpdir, f"{name}_roc.png")
                save_roc(y_test, y_score, roc_path)
                mlflow.log_artifact(roc_path, artifact_path="artifacts")

                # ✅ SAFE MODEL LOGGING (NO registry, NO name=)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                )

                print(
                    f"{name} → Accuracy={acc:.3f}, "
                    f"F1={f1:.3f}, ROC_AUC={roc:.3f}"
                )

        print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()