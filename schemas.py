from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    test_size: float = Field(0.25, ge=0.2, le=0.4)
    random_state: int = 42


class TrainResponse(BaseModel):
    message: str
    rows_used: int
    train_rows: int
    test_rows: int
    test_accuracy: float
    best_C: float
    cv_score: float


class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    attendance: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    hours_studied: float
    attendance: float
    passed: bool
    pass_probability: float


class EvaluateResponse(BaseModel):
    test_rows: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float