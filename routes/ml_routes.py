import logging

from fastapi import APIRouter, Depends, HTTPException

from models import ml_model
from schemas import (
    EvaluateResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from utils.auth import require_admin

logger = logging.getLogger("ml_api")
router = APIRouter(tags=["ML"])


@router.post(
    "/train",
    response_model=TrainResponse,
    dependencies=[Depends(require_admin)],
)
def train(req: TrainRequest = TrainRequest()):
    try:
        result = ml_model.train_model(req.test_size, req.random_state)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Training failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail=f"Training error: {e}")
    return TrainResponse(message="Model trained successfully", **result)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("Predict request: %s", req.model_dump())
    try:
        result = ml_model.predict(req.hours_studied, req.attendance)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return PredictResponse(**req.model_dump(), **result)


@router.get("/evaluate", response_model=EvaluateResponse)
def evaluate():
    try:
        result = ml_model.evaluate()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected evaluation error")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")
    return EvaluateResponse(**result)
