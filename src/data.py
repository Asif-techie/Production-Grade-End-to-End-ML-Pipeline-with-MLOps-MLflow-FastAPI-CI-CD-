"""
src/data.py

Data acquisition (API-first), fallback to local CSV, cleaning, EDA with MLflow logging.

Features:
- API → local → UCI fallback loading
- Safe pandas operations (NO chained assignment)
- Print first 5 rows (raw & cleaned)
- EDA visualizations
- MLflow nested runs
- CI-safe & pandas 3.0 compatible
"""

import os
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# =========================
# Constants
# =========================
UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

LOCAL_DATA_PATH = "data/heart.csv"
EDA_DIR = "data/eda"

COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# =========================
# Data acquisition
# =========================
def download_from_uci(save_path: str = LOCAL_DATA_PATH) -> pd.DataFrame:
    """Download dataset from UCI and save locally."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("🌐 Downloading dataset from UCI...")

    resp = requests.get(UCI_DOWNLOAD_URL, timeout=15)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(save_path, header=None)
    df.columns = COLS

    print(f"✅ Dataset downloaded and saved to {save_path}")
    return df

def load_raw_df() -> pd.DataFrame:
    """
    Load raw dataset using priority:
    1) Local CSV (guaranteed for CI)
    2) ucimlrepo API
    3) UCI download
    """

    # 1️⃣ Always try local first (CI-safe)
    if os.path.exists(LOCAL_DATA_PATH):
        print("📂 Loading dataset from local CSV")
        df = pd.read_csv(LOCAL_DATA_PATH)
        df.columns = COLS
        return df

    # 2️⃣ Try ucimlrepo
    try:
        print("🔌 Trying ucimlrepo API...")
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=45)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = COLS

        os.makedirs(os.path.dirname(LOCAL_DATA_PATH), exist_ok=True)
        df.to_csv(LOCAL_DATA_PATH, index=False)

        print("✅ Loaded dataset from ucimlrepo")
        return df

    except Exception as e:
        print("⚠ ucimlrepo failed:", e)

    # 3️⃣ Final fallback: UCI download
    try:
        print("⬇ Downloading dataset from UCI")
        return download_from_uci(LOCAL_DATA_PATH)
    except Exception as e:
        raise RuntimeError(
            "❌ Data loading failed. "
            "No local CSV and no internet access."
        ) from e


# =========================
# Data cleaning
# =========================
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe:
    - Replace '?' with NaN
    - Convert all columns to numeric
    - Impute ca & thal with mode
    - Convert target to binary
    - Drop remaining NaNs
    """
    df = df.copy()

    # Replace missing marker
    df = df.replace("?", np.nan)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safe imputation (NO inplace=True)
    for col in ["ca", "thal"]:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)

    # Binary target conversion
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Drop remaining missing values
    df = df.dropna().reset_index(drop=True)

    return df

# =========================
# EDA & Visualization
# =========================
def perform_eda(df: pd.DataFrame, save_dir: str = EDA_DIR):
    """
    Perform EDA and log artifacts to MLflow (nested run).
    """
    os.makedirs(save_dir, exist_ok=True)
    print("📊 Performing EDA...")

    with mlflow.start_run(run_name="EDA", nested=True):
        mlflow.log_param("rows", df.shape[0])
        mlflow.log_param("columns", df.shape[1])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Histograms
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            df[col].hist(bins=20)
            plt.title(f"Histogram - {col}")
            path = os.path.join(save_dir, f"hist_{col}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(path, artifact_path="eda")

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm")
        plt.title("Correlation Heatmap")
        corr_path = os.path.join(save_dir, "correlation_heatmap.png")
        plt.savefig(corr_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(corr_path, artifact_path="eda")

        # Class balance
        plt.figure(figsize=(5, 4))
        df["target"].value_counts().sort_index().plot(kind="bar")
        plt.title("Target Class Balance")
        balance_path = os.path.join(save_dir, "class_balance.png")
        plt.savefig(balance_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(balance_path, artifact_path="eda")

    print("✅ EDA artifacts logged to MLflow")

# =========================
# High-level loader
# =========================
def load_heart_data(run_eda: bool = True, verbose: bool = True):
    """
    Load, clean, optionally visualize, and return X, y, df.
    """
    raw = load_raw_df()

    if verbose:
        print("\n📌 RAW DATA (first 5 rows):")
        print(raw.head())

    df = clean_df(raw)

    if verbose:
        print("\n🧹 CLEANED DATA (first 5 rows):")
        print(df.head())

    if run_eda:
        perform_eda(df)

    X = df.drop(columns=["target"])
    y = df["target"].copy()

    return X, y, df

# =========================
# Script entry point
# =========================
if __name__ == "__main__":
    mlflow.set_experiment("heart-disease-data")

    with mlflow.start_run(run_name="data_pipeline"):
        X, y, df = load_heart_data(run_eda=True, verbose=True)

        print("\n📊 Final dataset shape:", df.shape)
        print("\n🎯 Target distribution:")
        print(y.value_counts())