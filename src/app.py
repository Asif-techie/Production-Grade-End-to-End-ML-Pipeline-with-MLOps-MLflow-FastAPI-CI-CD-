import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction API")

MODEL_DIR = "models"
models = {}

# Load all models dynamically
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        model_name = file.replace(".pkl", "")
        models[model_name] = joblib.load(os.path.join(MODEL_DIR, file))

if not models:
    raise RuntimeError("❌ No models found in models/ directory")


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
    return {
        "status": "ok",
        "loaded_models": list(models.keys())
    }


@app.post("/predict/{model_name}")
def predict(model_name: str, data: HeartInput):

    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not available"
        )

    model = models[model_name]

    X = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]]

    prediction = model.predict(X)[0]
    return {
        "model": model_name,
        "prediction": int(prediction)
    }
