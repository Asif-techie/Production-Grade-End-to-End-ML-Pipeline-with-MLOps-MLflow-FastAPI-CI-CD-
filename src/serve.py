# src/serve.py
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Heart Disease Prediction API")

# Load the trained pipeline
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
model_data = joblib.load(MODEL_PATH)
pipeline = model_data["pipeline"]

class PredictRequest(BaseModel):
    instances: List[List[float]]

@app.get("/health")
def health():
    try:
        _ = pipeline.predict([[0]*13])
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return {"status": "error", "model_loaded": False}

@app.post("/predict")
def predict(req: PredictRequest):
    preds = pipeline.predict(req.instances).tolist()
    try:
        conf = pipeline.predict_proba(req.instances).max(axis=1).tolist()
    except AttributeError:
        conf = preds
    return {"predictions": preds, "confidence": conf}