"""
src/pipeline.py

Defines imbalanced pipelines and hyperparameter search spaces.
Imported by train.py.
"""

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# ---------------- CONFIG ---------------- #
RANDOM_STATE = 42
K_CHOICES = [5, 7, 10, 12]
# ---------------------------------------- #

# ---------------- PIPELINES ---------------- #
pipelines = {
    "LogisticRegression": ImbPipeline(
        steps=[
            ("select", SelectKBest(score_func=f_classif)),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]
    ),

    "RandomForest": ImbPipeline(
        steps=[
            ("select", SelectKBest(score_func=f_classif)),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]
    ),

    "XGBoost": ImbPipeline(
        steps=[
            ("select", SelectKBest(score_func=f_classif)),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]
    ),
}

# ---------------- HYPERPARAMETER SPACES ---------------- #
param_spaces = {
    "LogisticRegression": {
        "select__k": K_CHOICES,
        "clf__C": [0.01, 0.1, 1, 5, 10],
    },

    "RandomForest": {
        "select__k": K_CHOICES,
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [4, 6, None],
        "clf__min_samples_split": [2, 5],
    },

    "XGBoost": {
        "select__k": K_CHOICES,
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
}

# ---------------- SEARCH STRATEGY ---------------- #
search_type = {
    "LogisticRegression": GridSearchCV,
    "RandomForest": RandomizedSearchCV,
    "XGBoost": RandomizedSearchCV,
}
