from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging
import joblib
import numpy as np
import os
from prometheus_fastapi_instrumentator import Instrumentator

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Globals ------------------
model = None

# ------------------ FastAPI App ------------------
app = FastAPI(title="Heart Disease Prediction API")

# ------------------ Prometheus Metrics ------------------
Instrumentator(should_group_status_codes=False).instrument(app).expose(app)

# ------------------ Startup: Load Model Safely ------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        models_path = os.getenv("MODELS_PATH", "models")
        model_path = os.path.join(models_path, "RandomForest.pkl")

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        model = None

# ------------------ Logging Middleware ------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡ {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"⬅ {request.method} {request.url} | status={response.status_code}")
    return response

# ------------------ Input Schemas ------------------
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

class InstancesInput(BaseModel):
    instances: list[list[float]]

# ------------------ Routes ------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: HeartInput | InstancesInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Support CI payload
    if isinstance(data, InstancesInput):
        X = np.array(data.instances)
    else:
        X = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                       data.fbs, data.restecg, data.thalach, data.exang,
                       data.oldpeak, data.slope, data.ca, data.thal]])

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    return {
        "prediction": int(pred[0]),
        "probability": float(proba[0])
    }
