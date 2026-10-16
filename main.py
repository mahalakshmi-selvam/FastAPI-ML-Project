from fastapi import FastAPI

from routes.ml_routes import router
from utils.logger import setup_logging

setup_logging()

app = FastAPI(
    title="ML API with FastAPI",
    description="Train, predict and evaluate a simple ML model.",
    version="3.0",
)

app.include_router(router)


@app.get("/")
def home():
    return {"message": "My ML API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}