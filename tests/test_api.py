import os
import joblib
import tempfile
from fastapi.testclient import TestClient
from src.serve import app

# create temporary model artifact for the API test
def make_temp_model(tmpdir):
    from sklearn.dummy import DummyClassifier
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = DummyClassifier(strategy="most_frequent")
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    # fit on tiny data
    pipeline.fit([[0],[1],[0],[1]],[0,1,0,1])

    data = {"pipeline": pipeline}
    path = os.path.join(tmpdir, "model.pkl")
    joblib.dump(data, path)
    return path

def test_predict_endpoint(tmp_path):
    model_path = make_temp_model(str(tmp_path))
    # set env to point to temp model
    os.environ["MODEL_PATH"] = model_path

    # reload app module to pick up new MODEL path (import side-effect)
    from importlib import reload
    import src.serve as serve_mod
    reload(serve_mod)

    client = TestClient(serve_mod.app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True

    r2 = client.post("/predict", json={"instances": [[0],[1]]})
    assert r2.status_code == 200
    assert "predictions" in r2.json()
