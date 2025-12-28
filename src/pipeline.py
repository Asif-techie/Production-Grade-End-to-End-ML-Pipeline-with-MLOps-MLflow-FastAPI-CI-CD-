# src/pipeline.py

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Define all pipelines in a dictionary
pipelines = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

# Preprocessing for numerical features (example)
numeric_transformer = StandardScaler()
numeric_imputer = SimpleImputer(strategy="mean")

# Example column transformer (replace with your columns)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, ["age", "trestbps", "chol", "thalach", "oldpeak"]),
    ]
)

# SMOTE example
smote = SMOTE(random_state=42)
