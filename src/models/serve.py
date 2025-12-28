import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)["pipeline"]

class PredictRequest(BaseModel):
    instances: list

@app.get("/health")
def health():
    return {"model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.instances)
    preds = model.predict(X)
    probs = model.predict_proba(X).max(axis=1)

    return {
        "predictions": preds.tolist(),
        "confidence": probs.tolist()
    }
