from fastapi import FastAPI
import joblib
import os
import logging

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Model path (single source of truth)
# -------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.pkl")

model = None


# -------------------------------------------------
# Load model on startup
# -------------------------------------------------
@app.on_event("startup")
def load_model():
    global model

    logger.info(f"Loading model from: {MODEL_PATH}")

    if not os.path.isfile(MODEL_PATH):
        logger.error(f"Model file NOT found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully")


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(payload: dict):
    if model is None:
        return {"error": "Model not loaded"}

    instances = payload.get("instances")
    if instances is None:
        return {"error": "Missing 'instances' in request body"}

    preds = model.predict(instances).tolist()

    # Handle models without predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(instances).tolist()
    else:
        probs = None

    return {
        "prediction": preds,
        "probability": probs
    }