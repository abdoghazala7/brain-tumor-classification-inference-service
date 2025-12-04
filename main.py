import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sentry_sdk

import config
import model_loader
import image_utils
import predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn= sentry_dsn,
        send_default_pii=True,
        enable_logs=True,
        traces_sample_rate=1.0,
        profile_session_sample_rate=1.0,
        profile_lifecycle="trace",
    )
    logger.info("✅ Sentry integration enabled.")
else:
    logger.warning("⚠️ Sentry DSN not found. Monitoring is disabled.")
    

MAX_FILE_SIZE = 5 * 1024 * 1024

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence_scores: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    message: str

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔄 Startup: Loading Deep Learning Model...")
    try:
        model = model_loader.load_model()
        ml_models["brain_tumor_model"] = model
        logger.info("✅ Startup: Model loaded successfully and ready for inference.")
    except Exception as e:
        logger.critical(f"❌ CRITICAL STARTUP ERROR: Failed to load model. Reason: {e}")
        sys.exit(1)
    yield
    ml_models.clear()
    logger.info("🛑 Shutdown: Model cleared from memory.")

app = FastAPI(
    title="Brain Tumor MRI Classification API",
    description="Production-ready API for classifying brain tumors using EfficientNetB0.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "message": "Brain Tumor Classifier API is active."
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_image(file: UploadFile = File(...)):
    allowed_types = [
        "image/jpeg", "image/png", "image/jpg", 
        "image/bmp", "image/gif", "image/tiff", "image/webp"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Supported types are: JPEG, PNG, BMP, GIF, TIFF, WEBP."
        )

    try:
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
             raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is too large. Max limit is {MAX_FILE_SIZE/(1024*1024)} MB."
            )

        image_tensor = image_utils.preprocess_image(contents)

        if image_tensor is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image processing failed. The file might be corrupted or not a valid image."
            )

        model = ml_models.get("brain_tumor_model")
        if model is None:
            logger.error("Model not found in memory during request.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded yet. Please try again later."
            )

        predicted_class, probabilities = predictor.predict(model, image_tensor)
        
        logger.info(f"Prediction successful for file: {file.filename} -> {predicted_class}")

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence_scores": probabilities
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred processing your request."
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)