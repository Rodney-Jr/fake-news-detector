import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib, os, argparse

def load_data(path: str):
    df = pd.read_csv(path)
    df["content"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    return df["content"], df["label"].astype(int)

def train(input_csv, out_dir):
    X, y = load_data(input_csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100_000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    print(classification_report(yte, yhat, digits=4))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))
    print("Saved model to", os.path.join(out_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", default="models/baseline")
    args = parser.parse_args()
    train(args.input_csv, args.out_dir)
