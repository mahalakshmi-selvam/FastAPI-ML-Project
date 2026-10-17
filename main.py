from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from routes.auth_routes import router as auth_router
from routes.ml_routes import router as ml_router
from routes.upload_routes import router as upload_router
from utils.auth import get_current_user, require_admin
from utils.logger import setup_logging
from utils.rate_limit import rate_limit

setup_logging()

app = FastAPI(
    title="ML API with FastAPI",
    description="Secure ML API: login, train, predict, evaluate.",
    version="4.0",
    dependencies=[Depends(rate_limit)],
)

app.include_router(auth_router)
app.include_router(ml_router, dependencies=[Depends(get_current_user)])
app.include_router(upload_router, dependencies=[Depends(require_admin)])

# Day 25: monitoring (Prometheus metrics at /metrics)
Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {"message": "My ML API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}