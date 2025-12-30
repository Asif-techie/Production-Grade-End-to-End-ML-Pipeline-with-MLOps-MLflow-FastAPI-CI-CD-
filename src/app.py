import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/HeartDisease_Models/Production"

model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="Heart Disease Prediction API")

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: HeartInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
