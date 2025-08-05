from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import mlflow.pyfunc
import pandas as pd
import os
import logging

app = FastAPI(title="Tourism Recommender API")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Set AWS credentials for S3
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")  
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        
        model = mlflow.pyfunc.load_model("models:/tourism-recommender-model@production")
        logging.info("Tourism model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")

class TourismInput(BaseModel):
    user_id: str

@app.post("/predict")
def predict(users: List[TourismInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    df = pd.DataFrame([{"user_id": user.user_id} for user in users])
    predictions = model.predict(df)
    return {"recommendations": predictions}