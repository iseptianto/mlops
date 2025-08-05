# model/model_tourism/register_model.py - FIXED VERSION
import os
import logging
import mlflow
import mlflow.pyfunc
import pickle
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)

class TourismRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load model artifacts"""
        try:
            self.user_encoder = pickle.load(open(context.artifacts["user_encoder"], "rb"))
            self.place_encoder = pickle.load(open(context.artifacts["place_encoder"], "rb"))
            self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix"], "rb"))
            self.content_similarity = pickle.load(open(context.artifacts["content_similarity"], "rb"))
            
            logging.info("✅ All model artifacts loaded successfully")
            logging.info(f"  - Users: {len(self.user_encoder)}")
            logging.info(f"  - Places: {len(self.place_encoder)}")
            logging.info(f"  - Prediction matrix: {self.prediction_matrix.shape}")
            logging.info(f"  - Content similarity: {self.content_similarity.shape}")
            
        except Exception as e:
            logging.error(f"❌ Error loading model artifacts: {e}")
            raise

    def predict(self, context, model_input):
        """Generate recommendations for users"""
        
        if "user_id" not in model_input.columns:
            raise ValueError("Input DataFrame must contain 'user_id' column")
        
        recommendations = []
        
        for user_id in model_input["user_id"]:
            try:
                if user_id not in self.user_encoder:
                    recommendations.append({
                        "user_id": user_id,
                        "recommendations": [],
                        "status": "unknown_user"
                    })
                    continue

                # Get user index
                user_idx = self.user_encoder[user_id]
                
                # Get user preferences
                user_scores = np.array(self.prediction_matrix[user_idx])
                
                # Get top 5 recommendations
                top_indices = user_scores.argsort()[-5:][::-1]
                
                # Format recommendations with scores
                user_recommendations = []
                for idx in top_indices:
                    place_name = self.place_encoder[idx]
                    score = float(user_scores[idx])
                    user_recommendations.append({
                        "place": place_name,
                        "score": round(score, 3),
                        "place_id": int(idx)
                    })
                
                recommendations.append({
                    "user_id": user_id,
                    "recommendations": user_recommendations,
                    "status": "success"
                })
                
            except Exception as e:
                logging.error(f"Error generating recommendations for user {user_id}: {e}")
                recommendations.append({
                    "user_id": user_id,
                    "recommendations": [],
                    "status": f"error: {str(e)}"
                })
        
        return recommendations

def check_required_files():
    """Check if all required pickle files exist"""
    required_files = [
        "user_encoder.pkl",
        "place_encoder.pkl", 
        "prediction_matrix.pkl",
        "content_similarity.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"❌ Missing required files: {missing_files}")
        logging.info("💡 Run 'python generate_tourism_data.py' first to generate missing files")
        return False
    
    logging.info("✅ All required files found")
    return True

def validate_model_artifacts():
    """Validate that model artifacts are valid"""
    try:
        # Load and validate each artifact
        with open("user_encoder.pkl", 'rb') as f:
            user_encoder = pickle.load(f)
        with open("place_encoder.pkl", 'rb') as f:
            place_encoder = pickle.load(f)
        with open("prediction_matrix.pkl", 'rb') as f:
            prediction_matrix = pickle.load(f)
        with open("content_similarity.pkl", 'rb') as f:
            content_similarity = pickle.load(f)
        
        # Validation checks
        assert isinstance(user_encoder, dict), "user_encoder must be a dictionary"
        assert isinstance(place_encoder, dict), "place_encoder must be a dictionary"
        assert isinstance(prediction_matrix, np.ndarray), "prediction_matrix must be numpy array"
        assert isinstance(content_similarity, np.ndarray), "content_similarity must be numpy array"
        
        # Shape consistency checks
        n_users = len(user_encoder)
        n_places = len(place_encoder)
        
        assert prediction_matrix.shape == (n_users, n_places), f"prediction_matrix shape mismatch: expected ({n_users}, {n_places}), got {prediction_matrix.shape}"
        assert content_similarity.shape == (n_places, n_places), f"content_similarity shape mismatch: expected ({n_places}, {n_places}), got {content_similarity.shape}"
        
        logging.info("✅ Model artifacts validation passed")
        return True
        
    except Exception as e:
        logging.error(f"❌ Model artifacts validation failed: {e}")
        return False

def register_tourism_model():
    """Register tourism recommendation model to MLflow"""
    
    # Set MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5001")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set AWS credentials
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    
    logging.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set experiment
    experiment_name = "tourism_recommender"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Tourism Model Registration") as run:
        run_id = run.info.run_id
        logging.info(f"Started MLflow run: {run_id}")
        
        # Log parameters
        params = {
            "model_type": "collaborative_filtering",
            "recommendation_algorithm": "user_based",
            "top_k": 5,
            "similarity_metric": "cosine"
        }
        mlflow.log_params(params)
        
        # Calculate and log metrics
        with open("prediction_matrix.pkl", 'rb') as f:
            prediction_matrix = pickle.load(f)
        with open("user_encoder.pkl", 'rb') as f:
            user_encoder = pickle.load(f)
        with open("place_encoder.pkl", 'rb') as f:
            place_encoder = pickle.load(f)
        
        metrics = {
            "num_users": len(user_encoder),
            "num_places": len(place_encoder),
            "matrix_density": np.count_nonzero(prediction_matrix) / prediction_matrix.size,
            "avg_user_rating": float(np.mean(prediction_matrix)),
            "model_coverage": 1.0  # All places can be recommended
        }
        mlflow.log_metrics(metrics)
        logging.info(f"Logged metrics: {metrics}")
        
        # Log model with artifacts
        logging.info("Logging PyFunc model...")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TourismRecommender(),
            artifacts={
                "user_encoder": "user_encoder.pkl",
                "place_encoder": "place_encoder.pkl",
                "prediction_matrix": "prediction_matrix.pkl",
                "content_similarity": "content_similarity.pkl"
            },
            conda_env="conda_env.yaml"
        )
        
        # Register model in Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_name = "tourism-recommender-model"
        
        client = MlflowClient()
        
        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(model_name)
            logging.info(f"Created new registered model: {model_name}")
        except Exception:
            logging.info(f"Model {model_name} already exists, continuing...")
        
        # Create model version
        model_version = client.create_model_version(
            name=model_name, 
            source=model_uri, 
            run_id=run_id
        )
        
        # Set alias to production
        client.set_registered_model_alias(
            name=model_name, 
            alias="production", 
            version=model_version.version
        )
        
        logging.info(f"✅ Model version {model_version.version} registered as '{model_name}@production'")
        
        return model_version

def test_registered_model():
    """Test the registered model"""
    try:
        logging.info("🧪 Testing registered model...")
        
        # Load model
        model = mlflow.pyfunc.load_model("models:/tourism-recommender-model@production")
        
        # Test prediction
        test_df = pd.DataFrame({"user_id": ["user_001", "user_002", "user_999"]})
        predictions = model.predict(test_df)
        
        logging.info("✅ Model test successful!")
        logging.info(f"Sample predictions: {predictions[:2]}")  # Show first 2 predictions
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    logging.info("🚀 Starting tourism model registration...")
    
    # Check required files
    if not check_required_files():
        logging.error("❌ Missing required files. Please run generate_tourism_data.py first.")
        exit(1)
    
    # Validate artifacts
    if not validate_model_artifacts():
        logging.error("❌ Invalid model artifacts.")
        exit(1)
    
    # Register model
    try:
        model_version = register_tourism_model()
        logging.info(f"✅ Model registration successful: version {model_version.version}")
        
        # Test registered model
        if test_registered_model():
            logging.info("🎉 Tourism recommendation model is ready!")
        else:
            logging.warning("⚠️ Model registered but test failed")
            
    except Exception as e:
        logging.error(f"❌ Model registration failed: {e}")
        exit(1)