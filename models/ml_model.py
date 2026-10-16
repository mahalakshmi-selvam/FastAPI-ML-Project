import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ml_api")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURES = ["hours_studied", "attendance"]
TARGET = "passed"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.dropna()
    return df[FEATURES].values, df[TARGET].values


def split_data(X, y, test_size, random_state):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Call POST /train first.")
    return joblib.load(MODEL_PATH)


def train_model(test_size: float = 0.25, random_state: int = 42) -> dict:
    logger.info("Training started (test_size=%s)", test_size)
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    test_accuracy = float(model.score(X_test, y_test))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "test_size": test_size, "random_state": random_state},
        MODEL_PATH,
    )
    logger.info("Training done. test_accuracy=%.2f", test_accuracy)

    return {
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": round(test_accuracy, 3),
    }


def predict(hours_studied: float, attendance: float) -> dict:
    model = load_bundle()["model"]
    x = np.array([[hours_studied, attendance]])
    label = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][1])
    return {"passed": bool(label), "pass_probability": round(prob, 3)}


def evaluate() -> dict:
    bundle = load_bundle()
    model = bundle["model"]
    X, y = load_data()
    _, X_test, _, y_test = split_data(
        X, y, bundle["test_size"], bundle["random_state"]
    )
    y_pred = model.predict(X_test)
    return {
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
    }