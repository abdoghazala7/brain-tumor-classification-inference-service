import sys
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import model_loader
import image_utils
import predictor

# Configure logging to capture events for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants for Validation
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit per image

# Response Schemas (Pydantic) 
class PredictionResponse(BaseModel):
    """
    Defines the standard structure for the API prediction response.
    """
    filename: str
    prediction: str
    confidence_scores: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    message: str


# Lifespan Manager (Model Loading) 
# Global dictionary to hold the model instance in memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the application.
    
    Startup: Loads the ML model. If loading fails, the app shuts down immediately (Fail Fast).
    Shutdown: Cleans up resources.
    """
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

# Essential for allowing frontend apps (React, Vue, etc.) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# - Endpoints -

@app.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    """
    Health Check Endpoint.
    Used by load balancers or Docker to verify the service is up and running.
    """
    return {
        "status": "healthy",
        "message": "Brain Tumor Classifier API is active."
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_image(file: UploadFile = File(...)):
    """
    Main inference endpoint.
    Validates, processes, and classifies the uploaded MRI image.
    """
    
    # A. Validate File Type
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
        # B. Read File & Validate Size
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
             raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is too large. Max limit is {MAX_FILE_SIZE/(1024*1024)} MB."
            )

        # C. Preprocess
        image_tensor = image_utils.preprocess_image(contents)

        if image_tensor is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image processing failed. The file might be corrupted or not a valid image."
            )

        # D. Retrieve Model
        model = ml_models.get("brain_tumor_model")
        if model is None:
            logger.error("Model not found in memory during request.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded yet. Please try again later."
            )

        # E. Inference
        predicted_class, probabilities = predictor.predict(model, image_tensor)
        
        logger.info(f"Prediction successful for file: {file.filename} -> {predicted_class}")

        # F. Return Response
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)