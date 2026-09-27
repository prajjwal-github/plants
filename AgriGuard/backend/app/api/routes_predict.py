from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Any
import shutil
import os
from pathlib import Path
import sys

# Add root to python path to access agriguard package
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.schemas.predict import AnalysisResponse
from agriguard.utils.config import load_config
from agriguard.inference.pipeline import InferencePipeline

router = APIRouter()

# Initialize pipeline
try:
    config = load_config("../configs/config.yaml")
    pipeline = InferencePipeline(config)
except Exception as e:
    print(f"Warning: Failed to load config/pipeline: {e}")
    pipeline = None

UPLOAD_DIR = Path("../outputs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Inference pipeline not initialized.")
        
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = pipeline.analyze(str(file_path))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
