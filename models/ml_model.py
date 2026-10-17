import io
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

logger = logging.getLogger("ml_api")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample.csv"
UPLOAD_PATH = BASE_DIR / "data" / "uploaded.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURES = ["hours_studied", "attendance"]
TARGET = "passed"


def get_data_path() -> Path:
    return UPLOAD_PATH if UPLOAD_PATH.exists() else DATA_PATH


def check_columns(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def save_dataset(content: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise ValueError("Invalid CSV file")
    check_columns(df)
    df = df[FEATURES + [TARGET]].dropna()
    if len(df) < 20:
        raise ValueError("Dataset needs at least 20 rows")
    df.to_csv(UPLOAD_PATH, index=False)
    return {"rows": int(len(df)), "columns": list(df.columns)}


def load_data():
    path = get_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    check_columns(df)
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

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid={"C": [0.01, 0.1, 1, 10, 100]},
        cv=3,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    test_accuracy = float(model.score(X_test, y_test))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "test_size": test_size, "random_state": random_state},
        MODEL_PATH,
    )
    logger.info("Training done. best_C=%s, test_accuracy=%.2f",
                grid.best_params_["C"], test_accuracy)

    return {
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": round(test_accuracy, 3),
        "best_C": float(grid.best_params_["C"]),
        "cv_score": round(float(grid.best_score_), 3),
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