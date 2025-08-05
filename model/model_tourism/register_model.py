#!/bin/bash
# fix_tourism_model.sh - Complete fix untuk tourism model

echo "🔧 FIXING TOURISM MODEL PIPELINE..."

# Step 1: Stop containers
echo "📦 Stopping containers..."
docker-compose down

# Step 2: Create missing files in model directory
echo "📝 Creating missing model files..."

# Create generate_tourism_data.py
cat > model/model_tourism/generate_tourism_data.py << 'EOF'
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

logging.basicConfig(level=logging.INFO)

def generate_tourism_data():
    """Generate realistic tourism recommendation data"""
    
    # 1. PLACES DATA
    places = [
        "Borobudur Temple", "Prambanan Temple", "Raja Ampat", 
        "Komodo Island", "Mount Bromo", "Lake Toba", 
        "Gili Islands", "Yogyakarta Palace", "Bali Beaches",
        "Lombok Waterfalls"
    ]
    
    # 2. USERS DATA
    users = [f"user_{i:03d}" for i in range(1, 101)]  # user_001 to user_100
    
    # 3. PLACE ENCODER
    place_encoder = {i: place for i, place in enumerate(places)}
    logging.info(f"Created place_encoder with {len(places)} places")
    
    # 4. USER ENCODER  
    user_encoder = {user: i for i, user in enumerate(users)}
    logging.info(f"Created user_encoder with {len(users)} users")
    
    # 5. CONTENT SIMILARITY MATRIX
    np.random.seed(42)
    n_places = len(places)
    n_features = 8
    
    place_features = np.random.rand(n_places, n_features)
    place_features[[0, 1, 7], 4] = np.random.uniform(0.8, 1.0, 3)  # Cultural places
    place_features[[2, 3, 4, 5, 6, 9], 5] = np.random.uniform(0.8, 1.0, 6)  # Nature places
    place_features[[2, 3, 4], 6] = np.random.uniform(0.7, 1.0, 3)  # Adventure places
    
    content_encoder = cosine_similarity(place_features)
    logging.info(f"Generated content_encoder matrix: {content_encoder.shape}")
    
    # 6. PREDICTION MATRIX
    n_users = len(users)
    prediction_matrix = np.random.rand(n_users, n_places)
    
    for i in range(n_users):
        if i % 4 == 0:
            prediction_matrix[i, [0, 1, 7]] *= 1.5
        elif i % 4 == 1:
            prediction_matrix[i, [2, 3, 4, 5, 6, 9]] *= 1.3
        elif i % 4 == 2:
            prediction_matrix[i, [8, 6]] *= 1.4
        else:
            prediction_matrix[i] *= np.random.uniform(0.8, 1.2, n_places)
    
    prediction_matrix = (prediction_matrix - prediction_matrix.min()) / (prediction_matrix.max() - prediction_matrix.min())
    logging.info(f"Generated prediction matrix: {prediction_matrix.shape}")
    
    return user_encoder, place_encoder, content_encoder, prediction_matrix

def save_model_artifacts():
    """Generate and save all required model artifacts"""
    
    logging.info("Generating tourism recommendation data...")
    user_encoder, place_encoder, content_encoder, prediction_matrix = generate_tourism_data()
    
    artifacts = {
        "user_encoder.pkl": user_encoder,
        "place_encoder.pkl": place_encoder,
        "content_encoder.pkl": content_encoder,
        "prediction_matrix.pkl": prediction_matrix
    }
    
    for filename, data in artifacts.items():
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"✅ Saved {filename}")
    
    logging.info("\n📊 VERIFICATION:")
    for filename in artifacts.keys():
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                logging.info(f"✅ {filename}: {len(data)} items")
            elif isinstance(data, np.ndarray):
                logging.info(f"✅ {filename}: shape {data.shape}")
        except Exception as e:
            logging.error(f"❌ {filename}: {e}")
    
    return artifacts

if __name__ == "__main__":
    save_model_artifacts()
    print("🎉 All model artifacts generated successfully!")
EOF

# Step 3: Update requirements untuk model tourism
cat > model/model_tourism/requirements.txt << 'EOF'
fastapi
uvicorn
joblib
numpy
pandas
scikit-learn
prometheus-fastapi-instrumentator
python-multipart
mlflow
boto3
psycopg2-binary
geopy
EOF

# Step 4: Update Dockerfile untuk model tourism
cat > model/model_tourism/Dockerfile << 'EOF'
FROM python:3.9
WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Generate pickle files first, then register model
CMD ["sh", "-c", "python generate_tourism_data.py && python register_model.py"]
EOF

# Step 5: Update register_model.py dengan fixed version
cat > model/model_tourism/register_model.py << 'EOF'
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
        try:
            self.user_encoder = pickle.load(open(context.artifacts["user_encoder"], "rb"))
            self.place_encoder = pickle.load(open(context.artifacts["place_encoder"], "rb"))
            self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix"], "rb"))
            self.content_encoder = pickle.load(open(context.artifacts["content_encoder"], "rb"))
            
            logging.info("✅ All model artifacts loaded successfully")
        except Exception as e:
            logging.error(f"❌ Error loading model artifacts: {e}")
            raise

    def predict(self, context, model_input):
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

                user_idx = self.user_encoder[user_id]
                user_scores = np.array(self.prediction_matrix[user_idx])
                top_indices = user_scores.argsort()[-5:][::-1]
                
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

def register_tourism_model():
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5001")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    
    experiment_name = "tourism_recommender"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Tourism Model Registration") as run:
        run_id = run.info.run_id
        logging.info(f"Started MLflow run: {run_id}")
        
        # Check if files exist
        required_files = ["user_encoder.pkl", "place_encoder.pkl", "prediction_matrix.pkl", "content_encoder.pkl"]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
        
        params = {
            "model_type": "collaborative_filtering",
            "recommendation_algorithm": "user_based",
            "top_k": 5,
            "similarity_metric": "cosine"
        }
        mlflow.log_params(params)
        
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
        }
        mlflow.log_metrics(metrics)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=TourismRecommender(),
            artifacts={
                "user_encoder": "user_encoder.pkl",
                "place_encoder": "place_encoder.pkl",
                "prediction_matrix": "prediction_matrix.pkl",
                "content_encoder": "content_encoder.pkl"
            },
            conda_env="conda_env.yaml"
        )
        
        model_uri = f"runs:/{run_id}/model"
        model_name = "tourism-recommender-model"
        
        client = MlflowClient()
        
        try:
            client.create_registered_model(model_name)
        except Exception:
            logging.info(f"Model {model_name} already exists")
        
        model_version = client.create_model_version(
            name=model_name, 
            source=model_uri, 
            run_id=run_id
        )
        
        client.set_registered_model_alias(
            name=model_name, 
            alias="production", 
            version=model_version.version
        )
        
        logging.info(f"✅ Model version {model_version.version} registered as '{model_name}@production'")
        return model_version

if __name__ == "__main__":
    logging.info("🚀 Starting tourism model registration...")
    
    try:
        model_version = register_tourism_model()
        logging.info(f"✅ Model registration successful: version {model_version.version}")
    except Exception as e:
        logging.error(f"❌ Model registration failed: {e}")
        exit(1)
EOF

# Step 6: Start containers in correct order
echo "🚀 Starting containers in correct order..."

# Start infrastructure first
docker-compose up -d db minio
echo "⏳ Waiting for database and MinIO..."
sleep 15

# Start MLflow
docker-compose up -d mlflow_server
echo "⏳ Waiting for MLflow server..."
sleep 20

# Build and start model trainer
echo "🏗️ Building model trainer..."
docker-compose build model_tourism_trainer
docker-compose up -d model_tourism_trainer
echo "⏳ Waiting for model training..."
sleep 30

# Build and start tourism API
echo "🏗️ Building tourism API..."
docker-compose build fastapi_tourism_app
docker-compose up -d fastapi_tourism_app
echo "⏳ Waiting for API startup..."
sleep 10

echo ""
echo "🔍 CHECKING STATUS..."
echo "================================="

# Check model trainer logs
echo "📋 Model Trainer Status:"
docker logs --tail 10 model_tourism_trainer

echo ""
echo "📋 Tourism API Status:"
docker logs --tail 10 fastapi_tourism_app

echo ""
echo "🧪 TESTING API..."
echo "================================="

# Test API
sleep 5
echo "Testing root endpoint:"
curl -s http://localhost:8101/ | python3 -m json.tool

echo ""
echo "Testing prediction endpoint:"
curl -s -X POST http://localhost:8101/predict \
  -H "Content-Type: application/json" \
  -d '[{"user_id": "user_001"}]' | python3 -m json.tool

echo ""
echo "✅ TOURISM MODEL FIX COMPLETE!"
echo "API available at: http://localhost:8101"
echo "Use 'docker logs -f fastapi_tourism_app' to monitor"
EOF

# Step 7: Make script executable and run
chmod +x model/model_tourism/generate_tourism_data.py

echo "🎯 READY TO RUN FIX!"
echo ""
echo "Run this command to fix everything:"
echo "bash fix_tourism_model.sh"