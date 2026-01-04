from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load the only model
models_dir = os.path.join(os.path.dirname(__file__), "models")
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
if not model_files:
    raise RuntimeError("No model found in models directory!")

model_path = os.path.join(models_dir, model_files[0])
model = joblib.load(model_path)
model_class = type(model).__name__
print(f"Using model: {model_class}")

# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
async def predict(payload: dict):
    # Expected feature order
    expected_features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                         "exang","oldpeak","slope","ca","thal"]
    
    try:
        # Extract features in the correct order
        X = np.array([[payload[f] for f in expected_features]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    
    try:
        pred = model.predict(X).tolist()
        prob = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None
        return JSONResponse({"model_used": model_class, "prediction": pred, "probability": prob})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
