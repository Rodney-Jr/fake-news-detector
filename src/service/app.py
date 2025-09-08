from fastapi import FastAPI
from src.inference.schemas import NewsItem, Prediction
from src.inference.pipeline import predict_one
import os

app = FastAPI(title="Fake News Detector")

@app.get("/health")
def health():
    # Check if model file exists
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        return {"status": "error", "details": "Model file not found"}

    try:
        # Try a dummy prediction
        _ = predict_one("Test title", "Test body")
    except Exception as e:
        return {"status": "error", "details": str(e)}

    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(item: NewsItem):
    label, proba = predict_one(item.title, item.text)
    return {"label": label, "proba": proba}

