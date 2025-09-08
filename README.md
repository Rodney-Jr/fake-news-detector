# Fake News Detector

End-to-end ML project for detecting fake news.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/training/train_sklearn.py --input_csv data/train.csv --out_dir models/baseline
uvicorn src.service.app:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.
