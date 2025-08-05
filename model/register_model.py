# Exec ke MLflow container
docker exec -it mlflow_server bash

# Di dalam container, buat register script
cat > /tmp/register_tourism.py << 'EOF'
import os
import mlflow
import mlflow.pyfunc
import pickle
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)

os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'

mlflow.set_tracking_uri('http://localhost:5001')

class TourismRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.user_encoder = pickle.load(open(context.artifacts["user_encoder"], "rb"))
        self.place_encoder = pickle.load(open(context.artifacts["place_encoder"], "rb"))
        self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix"], "rb"))
        self.content_similarity = pickle.load(open(context.artifacts["content_similarity"], "rb"))
        logging.info("✅ Model artifacts loaded successfully")

    def predict(self, context, model_input):
        recommendations = []
        for user_id in model_input["user_id"]:
            if user_id not in self.user_encoder:
                recommendations.append({"user_id": user_id, "recommendations": [], "status": "unknown_user"})
                continue
            
            user_idx = self.user_encoder[user_id]
            user_scores = np.array(self.prediction_matrix[user_idx])
            top_indices = user_scores.argsort()[-5:][::-1]
            
            user_recommendations = []
            for idx in top_indices:
                place_name = self.place_encoder[idx]
                score = float(user_scores[idx])
                user_recommendations.append({
                    "place": place_name,
                    "score": round(score, 3)
                })
            
            recommendations.append({
                "user_id": user_id,
                "recommendations": user_recommendations,
                "status": "success"
            })
        
        return recommendations

# Change to /tmp directory where pickle files are
os.chdir('/tmp')

# Check all files exist
required_files = ["user_encoder.pkl", "place_encoder.pkl", "prediction_matrix.pkl", "content_similarity.pkl"]
for file in required_files:
    if not os.path.exists(file):
        print(f"❌ Missing file: {file}")
        exit(1)
    else:
        print(f"✅ Found: {file}")

mlflow.set_experiment("tourism_recommender")

with mlflow.start_run(run_name="Manual Tourism Registration") as run:
    print(f"🚀 Starting MLflow run: {run.info.run_id}")
    
    # Log model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TourismRecommender(),
        artifacts={
            "user_encoder": "user_encoder.pkl",
            "place_encoder": "place_encoder.pkl",
            "prediction_matrix": "prediction_matrix.pkl",
            "content_similarity": "content_similarity.pkl"
        }
    )
    
    # Register model
    model_uri = f"runs:/{run.info.run_id}/model"
    client = MlflowClient()
    
    try:
        # Try to create registered model
        try:
            client.create_registered_model("tourism-recommender-model")
        except:
            print("Model already exists, continuing...")
        
        mv = client.create_model_version(
            name="tourism-recommender-model",
            source=model_uri,
            run_id=run.info.run_id
        )
        
        client.set_registered_model_alias(
            name="tourism-recommender-model",
            alias="production",
            version=mv.version
        )
        
        print(f"✅ Model registered successfully! Version: {mv.version}")
        
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        exit(1)
EOF

# Jalankan register script
cd /tmp
python3 register_tourism.py

# Exit dari container
exit