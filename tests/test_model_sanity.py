import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_model_can_fit_and_predict():
    X = np.array([[20], [30], [40], [50]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)