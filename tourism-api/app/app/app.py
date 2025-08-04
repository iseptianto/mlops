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
async def predict(request: Request):
    input_data = await request.json()  # Expecting [{"user_id": "user_001"}]
    input_df = pd.DataFrame(input_data)
    predictions = model.predict(input_df)
    return {"recommendations": predictions}
