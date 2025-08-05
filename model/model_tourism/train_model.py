import mlflow
import pickle

# Set experiment name
mlflow.set_experiment("tourism_recommender")

with mlflow.start_run():
    mlflow.log_param("model_type", "content_encoder_recommender")

    # Log each artifact (4 total)
    artifacts = {
        "content_encoder.pkl": "content_encoder.pkl",
        "place_encoder.pkl": "place_encoder.pkl",
        "prediction_matrix.pkl": "prediction_matrix.pkl",
        "user_encoder.pkl": "user_encoder.pkl"
    }

    for name, file_path in artifacts.items():
        mlflow.log_artifact(file_path, artifact_path="model_artifacts")

    print("✅ All artifacts logged to MLflow.")
