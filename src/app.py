from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# ---------------- MODELS ----------------
models_dir = os.path.join(os.path.dirname(__file__), "models")
if not os.path.exists(models_dir):
    raise RuntimeError(f"Models directory not found: {models_dir}")

# Load all models dynamically
models = {}
for file in os.listdir(models_dir):
    if file.endswith(".pkl"):
        model_name = file.replace(".pkl", "")
        model_path = os.path.join(models_dir, file)
        models[model_name] = joblib.load(model_path)

if not models:
    raise RuntimeError("No models found in models directory!")

print(f"✅ Loaded models: {list(models.keys())}")

# Set default model
DEFAULT_MODEL = "RandomForest" if "RandomForest" in models else list(models.keys())[0]

# ---------------- FRONTEND ----------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if not os.path.exists(frontend_path):
    raise RuntimeError(f"Frontend directory not found: {frontend_path}")

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# ---------------- API ----------------
@app.post("/predict")
async def predict(payload: dict):
    instances = payload.get("instances")
    model_name = payload.get("model_name", DEFAULT_MODEL)

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )

    if not instances:
        raise HTTPException(status_code=400, detail="No 'instances' provided")

    try:
        X = np.array(instances)
        model = models[model_name]
        preds = model.predict(X).tolist()
        probs = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None

        return {"model_used": model_name, "prediction": preds, "probability": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
