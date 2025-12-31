from fastapi import FastAPI
import joblib
import os

app = FastAPI()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/LogisticRegression.pkl"
)

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    instances = payload.get("instances")
    preds = model.predict(instances)
    probs = model.predict_proba(instances).tolist()
    return {
        "prediction": preds.tolist(),
        "probability": probs
    }
