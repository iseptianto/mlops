from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load tourism model
model = mlflow.pyfunc.load_model(model_uri="models:/tourism-recommender-model@production")


@app.get("/")
def read_root():
    return {"api_status": "ok", "model_name": "tourism-recommender-model", "model_alias": "production"}

@app.post("/predict")
def predict(payload: List[Dict[str, str]]):
    df = pd.DataFrame(payload)
    predictions = model.predict(df)
    return {"recommendations": predictions}

