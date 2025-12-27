import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import requests
import seaborn as sns

UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"
LOCAL_DATA_PATH = "data/heart.csv"
EDA_DIR = "data/eda"

COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_from_uci(save_path: str = LOCAL_DATA_PATH) -> pd.DataFrame:
    """Download UCI Cleveland dataset and save CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("🌐 Downloading dataset from UCI:", UCI_DOWNLOAD_URL)
    resp = requests.get(UCI_DOWNLOAD_URL, timeout=15)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(save_path, header=None)
    df.columns = COLS
    print(f"✅ Downloaded and saved to {save_path}")
    return df


def load_raw_df() -> pd.DataFrame:
    """Load dataset: API -> local CSV -> UCI download."""
    try:
        print("🔌 Trying ucimlrepo.fetch_ucirepo(id=45)...")
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=45)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        if df.shape[1] == 14:
            df.columns = COLS

        print("✅ Loaded dataset from ucimlrepo API.")
        os.makedirs(os.path.dirname(LOCAL_DATA_PATH), exist_ok=True)
        df.to_csv(LOCAL_DATA_PATH, index=False)
        return df
    except Exception as e:
        print("⚠ ucimlrepo load failed:", e)

    if os.path.exists(LOCAL_DATA_PATH):
        print("📂 Loading dataset from local CSV:", LOCAL_DATA_PATH)
        df = pd.read_csv(LOCAL_DATA_PATH, header=None)
        df.columns = COLS
        return df

    try:
        return download_from_uci(LOCAL_DATA_PATH)
    except Exception as e:
        raise RuntimeError(
            "Failed to obtain dataset from API, local file, and UCI download"
        ) from e


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: replace ?, convert numeric, impute, binary target."""
    df = df.copy()
    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("ca", "thal"):
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    if "target" in df.columns:
        df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    df = df.dropna().reset_index(drop=True)
    return df


def perform_eda(df: pd.DataFrame, save_dir: str = EDA_DIR):
    """Generate histograms, heatmap, class balance plots and log to MLflow."""
    os.makedirs(save_dir, exist_ok=True)
    print("📊 Performing EDA and logging to MLflow (nested run)...")

    with mlflow.start_run(run_name="EDA", nested=True):
        mlflow.log_param("eda_rows", df.shape[0])
        mlflow.log_param("eda_columns", df.shape[1])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            df[col].hist(bins=20)
            plt.title(f"Histogram - {col}")
            plot_path = os.path.join(save_dir, f"hist_{col}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="eda_plots")

        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            corr_path = os.path.join(save_dir, "correlation_heatmap.png")
            plt.savefig(corr_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(corr_path, artifact_path="eda_plots")

        if "target" in df.columns:
            plt.figure(figsize=(5, 4))
            df["target"].value_counts().sort_index().plot(kind="bar")
            plt.title("Target Class Balance (0=Healthy,1=Disease)")
            balance_path = os.path.join(save_dir, "class_balance.png")
            plt.savefig(balance_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(balance_path, artifact_path="eda_plots")

    print("✅ EDA artifacts logged to MLflow.")


def load_heart_data(run_eda: bool = True):
    """Load raw, clean, log previews, optional EDA, return X, y, df."""
    raw = load_raw_df()
    print("\n📌 RAW DATA (first 5 rows):")
    print(raw.head())

    os.makedirs("data/previews", exist_ok=True)
    raw_preview_path = "data/previews/raw_head.csv"
    raw.head().to_csv(raw_preview_path, index=False)
    mlflow.log_artifact(raw_preview_path, artifact_path="data_preview")

    df = clean_df(raw)
    print("\n🧹 CLEANED DATA (first 2 rows):")
    print(df.head(2))

    clean_preview_path = "data/previews/clean_head.csv"
    df.head(2).to_csv(clean_preview_path, index=False)
    mlflow.log_artifact(clean_preview_path, artifact_path="data_preview")

    if run_eda:
        perform_eda(df)

    X = df.drop(columns=["target"])
    y = df["target"].copy()

    return X, y, df
