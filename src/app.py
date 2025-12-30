from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Logging
logging.basicConfig(level=logging.INFO)

# Load model
MODEL_PATH = "models/LogisticRegression.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI()

# Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Middleware logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.post("/predict")
def predict(data: HeartInput):
    X = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal
    ]])
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {"prediction": int(pred[0]), "probability": float(proba[0])}
