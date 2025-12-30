"""
src/train.py

Production-grade training script:
- Runs EDA as nested MLflow run
- Trains multiple models with hyperparameter search
- Logs metrics & artifacts to MLflow
- Saves trained models locally for Docker deployment
"""
import matplotlib
matplotlib.use("Agg") 

import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
from math import prod

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pipeline import pipelines, param_spaces, search_type
from data import load_heart_data
from utils_plot import save_cm, save_roc

# ---------------- CONFIG ---------------- #
EXPERIMENT_NAME = "HeartDisease_Models"
mlflow.set_experiment(EXPERIMENT_NAME)

N_JOBS = -1
CV_FOLDS = 5
RANDOM_STATE = 42

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
# ---------------------------------------- #


def total_param_combinations(param_grid):
    """Calculate total number of parameter combinations."""
    sizes = [len(v) for v in param_grid.values()]
    return prod(sizes) if sizes else 1


def main():
    with mlflow.start_run(run_name="Main_Training_Run"):

        # Load data + run EDA (nested run)
        X, y, _ = load_heart_data(run_eda=True)

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])

        # ---------------- TRAIN MODELS ---------------- #
        for name, pipe in pipelines.items():
            print(f"\n🔹 Starting hyperparameter search for {name}")

            Searcher = search_type.get(name)
            params = param_spaces.get(name, {})

            if Searcher is None:
                print(f"⚠ No search strategy defined for {name}. Skipping.")
                continue

            if Searcher == GridSearchCV:
                search = Searcher(
                    pipe,
                    params,
                    cv=CV_FOLDS,
                    n_jobs=N_JOBS,
                    scoring="roc_auc",
                )
            else:
                max_combos = total_param_combinations(params)
                n_iter = min(25, max_combos)
                search = Searcher(
                    pipe,
                    params,
                    n_iter=n_iter,
                    cv=CV_FOLDS,
                    n_jobs=N_JOBS,
                    scoring="roc_auc",
                    random_state=RANDOM_STATE,
                )

            # -------- Nested MLflow run per model -------- #
            with mlflow.start_run(run_name=name, nested=True):

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Predictions
                y_pred = best_model.predict(X_test)
                try:
                    y_score = best_model.predict_proba(X_test)[:, 1]
                except Exception:
                    try:
                        y_score = best_model.decision_function(X_test)
                    except Exception:
                        y_score = y_pred

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_score)

                # Log params
                for k, v in search.best_params_.items():
                    mlflow.log_param(k, v)

                # Log metrics
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc)

                # -------- Save plots as artifacts -------- #
                tmpdir = tempfile.mkdtemp()

                cm_path = os.path.join(tmpdir, f"{name}_cm.png")
                save_cm(y_test, y_pred, cm_path)
                mlflow.log_artifact(cm_path, artifact_path="artifacts")

                roc_path = os.path.join(tmpdir, f"{name}_roc.png")
                save_roc(y_test, y_score, roc_path)
                mlflow.log_artifact(roc_path, artifact_path="artifacts")

                # -------- Save model locally (Docker needs this) -------- #
                model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
                joblib.dump(best_model, model_path)
                print(f"✅ Saved model locally → {model_path}")

                # -------- Log model to MLflow (NO `name=`) -------- #
                mlflow.sklearn.log_model(best_model, artifact_path=name)

                print(
                    f"✅ {name} | Accuracy={acc:.3f}, "
                    f"F1={f1:.3f}, ROC_AUC={roc:.3f}"
                )

        print("\n🎉 Training complete. Models saved + logged to MLflow.")


if __name__ == "__main__":
    main()
