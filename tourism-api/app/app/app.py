# File: tourism-api/app/app/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/tourism-recommender-model@production")

app = FastAPI()

# Define input schema
class UserInput(BaseModel):
    user_id: int

@app.get("/")
def read_root():
    return {"api_status": "ok", "model_name": "tourism-recommender-model"}

@app.post("/predict")
def get_recommendation(input_data: UserInput):
    try:
        # Create DataFrame
        df = pd.DataFrame({"user_id": [input_data.user_id]})

        # Run prediction
        result = model.predict(df)

        return {"user_id": input_data.user_id, "recommendations": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
