import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5001")  # atau http://mlflow_server:5001 jika dalam container

model = mlflow.pyfunc.load_model("models:/tourism-recommender-model@production")
df = pd.DataFrame({"user_id": [123]})
print(model.predict(df))
