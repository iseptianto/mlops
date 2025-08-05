import mlflow.pyfunc
import pickle
import numpy as np
class TourismRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        try:
            self.user_encoder = pickle.load(open(context.artifacts["user_encoder"], "rb"))
            self.place_encoder = pickle.load(open(context.artifacts["place_encoder"], "rb")) 
            self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix"], "rb"))
            self.content_encoder = pickle.load(open(context.artifacts["content_encoder"], "rb"))
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {e}")

    def predict(self, context, model_input):
        if "user_id" not in model_input.columns:
            raise ValueError("Input must contain 'user_id' column")
            
        recommendations = []
        for user in model_input["user_id"]:
            if user not in self.user_encoder:
                recommendations.append({"user_id": user, "recommendations": [], "status": "unknown_user"})
                continue

            user_idx = self.user_encoder[user]
            user_scores = np.array(self.prediction_matrix[user_idx])
            top_indices = user_scores.argsort()[-5:][::-1]  # Top 5
            top_places = [self.place_encoder[i] for i in top_indices]
            top_scores = [float(user_scores[i]) for i in top_indices]
            
            recommendations.append({
                "user_id": user,
                "recommendations": [
                    {"place": place, "score": score} 
                    for place, score in zip(top_places, top_scores)
                ],
                "status": "success"
            })
        
        return recommendations

