from fastapi import FastAPI, HTTPException
import joblib
import os
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Path to model inside container
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/LogisticRegression.pkl")

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    instances = payload.get("instances")
    if not instances:
        raise HTTPException(status_code=400, detail="No 'instances' provided")
    try:
        X = np.array(instances)
        preds = model.predict(X).tolist()
        probs = model.predict_proba(X).tolist()
        return {"prediction": preds, "probability": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
