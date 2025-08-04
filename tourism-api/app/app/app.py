# app.py
import pickle
from fastapi import FastAPI, Request
import pandas as pd

app = FastAPI()
model = pickle.load(open("models/tourism_model.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": pred.tolist()}
