from fastapi import FastAPI, Request
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

# Load model from MLflow registry alias
model = mlflow.pyfunc.load_model(model_uri="models:/tourism-recommender-model@production")

@app.get("/")
def root():
    return {"api_status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    input_data = await request.json()
    df = pd.DataFrame(input_data)

    # Jalankan prediksi
    results = model.predict(df)

    return {"recommendations": results}
