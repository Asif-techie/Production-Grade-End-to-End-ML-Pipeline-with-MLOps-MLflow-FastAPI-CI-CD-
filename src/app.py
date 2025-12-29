import joblib
import pandas as pd
from fastapi import FastAPI

MODEL_PATH = "artifacts/models/LogisticRegression.pkl"

app = FastAPI()
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
