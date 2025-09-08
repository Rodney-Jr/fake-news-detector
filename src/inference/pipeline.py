import joblib, os
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline/model.joblib")
_model = None
LABELS = {0: "real", 1: "fake"}

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_one(title: str, text: str):
    model = get_model()
    content = f"{title or ''} {text or ''}".strip()
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba([content])[0][1])
    pred = int(model.predict([content])[0])
    return LABELS[pred], proba
