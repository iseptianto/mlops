# model/model_tourism/generate_tourism_data.py
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import logging

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
    # Generate place features (location, type, price, rating, etc.)
    np.random.seed(42)
    n_places = len(places)
    n_features = 8  # lat, lng, price_level, rating, cultural, nature, adventure, family
    
    place_features = np.random.rand(n_places, n_features)
    
    # Add some realistic patterns
    # Cultural places (Borobudur, Prambanan, Yogyakarta Palace)
    place_features[[0, 1, 7], 4] = np.random.uniform(0.8, 1.0, 3)  # High cultural score
    
    # Nature places (Raja Ampat, Komodo, Bromo, Toba, Gili, Lombok)
    place_features[[2, 3, 4, 5, 6, 9], 5] = np.random.uniform(0.8, 1.0, 6)  # High nature score
    
    # Adventure places (Raja Ampat, Komodo, Bromo)
    place_features[[2, 3, 4], 6] = np.random.uniform(0.7, 1.0, 3)  # High adventure score
    
    content_encoder = cosine_similarity(place_features)
    logging.info(f"Generated content_encoder matrix: {content_encoder.shape}")
    
    # 6. PREDICTION MATRIX (User-Place Preferences)
    # Generate user preferences based on user profiles
    n_users = len(users)
    prediction_matrix = np.random.rand(n_users, n_places)
    
    # Add some realistic user behavior patterns
    for i in range(n_users):
        # Some users prefer cultural sites
        if i % 4 == 0:
            prediction_matrix[i, [0, 1, 7]] *= 1.5  # Boost cultural places
        # Some users prefer nature
        elif i % 4 == 1:
            prediction_matrix[i, [2, 3, 4, 5, 6, 9]] *= 1.3  # Boost nature places
        # Some users prefer beaches
        elif i % 4 == 2:
            prediction_matrix[i, [8, 6]] *= 1.4  # Boost beach places
        # Balanced users
        else:
            prediction_matrix[i] *= np.random.uniform(0.8, 1.2, n_places)
    
    # Normalize to 0-1 range
    prediction_matrix = (prediction_matrix - prediction_matrix.min()) / (prediction_matrix.max() - prediction_matrix.min())
    logging.info(f"Generated prediction matrix: {prediction_matrix.shape}")
    
    return user_encoder, place_encoder, content_encoder, prediction_matrix

def save_model_artifacts():
    """Generate and save all required model artifacts"""
    
    logging.info("Generating tourism recommendation data...")
    user_encoder, place_encoder, content_encoder, prediction_matrix = generate_tourism_data()
    
    # Save all pickle files
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
    
    # Verify files
    logging.info("\n📊 VERIFICATION:")
    for filename in artifacts.keys():
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                logging.info(f"✅ {filename}: {len(data)} items")
            elif isinstance(data, np.ndarray):
                logging.info(f"✅ {filename}: shape {data.shape}")
            else:
                logging.info(f"✅ {filename}: type {type(data)}")
        except Exception as e:
            logging.error(f"❌ {filename}: {e}")
    
    logging.info("\n🎉 All model artifacts generated successfully!")
    
    return artifacts

def test_model_artifacts():
    """Test if generated artifacts work correctly"""
    
    logging.info("\n🧪 TESTING MODEL ARTIFACTS...")
    
    try:
        # Load artifacts
        with open("user_encoder.pkl", 'rb') as f:
            user_encoder = pickle.load(f)
        with open("place_encoder.pkl", 'rb') as f:
            place_encoder = pickle.load(f)
        with open("prediction_matrix.pkl", 'rb') as f:
            prediction_matrix = pickle.load(f)
        with open("content_encoder.pkl", 'rb') as f:
            content_encoder = pickle.load(f)
        
        # Test recommendation logic
        test_user = "user_001"
        if test_user in user_encoder:
            user_idx = user_encoder[test_user]
            user_scores = prediction_matrix[user_idx]
            top_indices = user_scores.argsort()[-5:][::-1]
            top_places = [place_encoder[i] for i in top_indices]
            top_scores = [user_scores[i] for i in top_indices]
            
            logging.info(f"\n📍 TEST RECOMMENDATION for {test_user}:")
            for place, score in zip(top_places, top_scores):
                logging.info(f"  - {place}: {score:.3f}")
        
        logging.info("\n✅ Model artifacts test PASSED!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Model artifacts test FAILED: {e}")
        return False

if __name__ == "__main__":
    # Generate artifacts
    artifacts = save_model_artifacts()
    
    # Test artifacts
    test_model_artifacts()