from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)

# ------------------ Load Model ------------------
models_dir = "models"
model_path = f"{models_dir}/RandomForest.pkl"  # adjust to your default model
model = joblib.load(model_path)

# ------------------ FastAPI App ------------------
app = FastAPI(title="Heart Disease Prediction API")

# ------------------ Prometheus Metrics ------------------
instrumentator = Instrumentator(should_group_status_codes=False)
instrumentator.instrument(app).expose(app)

# ------------------ Logging Middleware ------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"➡ {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"⬅ {request.method} {request.url} | status={response.status_code}")
    return response

# ------------------ Input Schema ------------------
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

# ------------------ Routes ------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(data: HeartInput):
    X = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                   data.fbs, data.restecg, data.thalach, data.exang,
                   data.oldpeak, data.slope, data.ca, data.thal]])
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {"prediction": int(pred[0]), "probability": float(proba[0])}
