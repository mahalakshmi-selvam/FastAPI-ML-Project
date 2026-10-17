import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from models import ml_model

logger = logging.getLogger("ml_api")
router = APIRouter(tags=["Data"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed")
    content = await file.read()
    try:
        info = ml_model.save_dataset(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("Dataset uploaded: %s", info)
    return {"message": "Dataset uploaded successfully", **info}