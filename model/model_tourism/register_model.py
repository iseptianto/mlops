import mlflow
from mlflow.tracking import MlflowClient

# Connect ke MLflow
mlflow.set_tracking_uri("http://mlflow_server:5001")
experiment_name = "tourism_recommender"
mlflow.set_experiment(experiment_name)

# Start run
with mlflow.start_run(run_name="Register Tourism Model") as run:
    run_id = run.info.run_id

    # Log parameter model
    mlflow.log_param("use_case", "tourism_recommender")

    # Log artifacts
    mlflow.log_artifact("content_similarity.pkl", artifact_path="model_artifacts")
    mlflow.log_artifact("place_encoder.pkl", artifact_path="model_artifacts")
    mlflow.log_artifact("prediction_matrix.pkl", artifact_path="model_artifacts")
    mlflow.log_artifact("user_encoder.pkl", artifact_path="model_artifacts")

    print(f"🏃 View run at: http://mlflow_server:5001/#/experiments/{run.info.experiment_id}/runs/{run_id}")

    # Register model ke model registry
    model_uri = f"runs:/{run_id}/model_artifacts"
    model_name = "tourism-recommender-model"

    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception as e:
        print("⚠️ Model already registered. Continuing...")

    client.create_model_version(name=model_name, source=model_uri, run_id=run_id)

    print(f"✅ Model version created and registered as: {model_name}")
