import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Predictor")

# Load trained model
model = joblib.load("artifacts/RandomForest_model.pkl")  # Replace with your actual path


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    y_pred = model.predict(df)[0]
    y_prob = model.predict_proba(df)[0, 1] if hasattr(model, "predict_proba") else None
    return {"prediction": int(y_pred), "confidence": float(y_prob) if y_prob else None}
