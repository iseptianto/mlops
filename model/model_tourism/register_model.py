import mlflow.pyfunc
import pickle

class TourismRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.user_encoder = pickle.load(open(context.artifacts["user_encoder"], "rb"))
        self.place_encoder = pickle.load(open(context.artifacts["place_encoder"], "rb"))
        self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix"], "rb"))
        self.similarity = pickle.load(open(context.artifacts["content_similarity"], "rb"))

    def predict(self, context, model_input):
        # Implementasi prediksi rekomendasi di sini
        return ["Place A", "Place B"]

if __name__ == "__main__":
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("http://mlflow_server:5001")
    experiment_name = "tourism_recommender"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Register Tourism Model") as run:
        run_id = run.info.run_id

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TourismRecommender(),
            artifacts={
                "content_similarity": "content_similarity.pkl",
                "place_encoder": "place_encoder.pkl",
                "prediction_matrix": "prediction_matrix.pkl",
                "user_encoder": "user_encoder.pkl"
            }
            conda_env="conda_env.yaml"
        )


        model_uri = f"runs:/{run_id}/model"
        model_name = "tourism-recommender-model"

        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except Exception:
            print("⚠️ Model already registered. Continuing...")

        mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)

        client.set_registered_model_alias(name=model_name, alias="production", version=mv.version)

        print(f"✅ Model version {mv.version} registered as '{model_name}' with alias 'production'")
