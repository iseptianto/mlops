import mlflow.pyfunc
import pandas as pd
import pickle
import numpy as np

class TourismRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load model artifacts (pastikan nama file sama)
        self.similarity = pickle.load(open(context.artifacts["content_similarity.pkl"], "rb"))
        self.place_encoder = pickle.load(open(context.artifacts["place_encoder.pkl"], "rb"))
        self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix.pkl"], "rb"))
        self.user_encoder = pickle.load(open(context.artifacts["user_encoder.pkl"], "rb"))

        # Create reverse mapping for decoded output
        self.place_decoder = {v: k for k, v in self.place_encoder.items()}
        self.user_decoder = {v: k for k, v in self.user_encoder.items()}

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        recommendations = []

        for user_id in model_input["user_id"]:
            if user_id not in self.user_encoder:
                recommendations.append(["Unknown user"])
                continue

            user_idx = self.user_encoder[user_id]
            user_ratings = self.prediction_matrix[user_idx]

            # Ambil 5 rekomendasi teratas
            top_indices = np.argsort(user_ratings)[::-1][:5]

            # Decode place_id
            top_places = [self.place_decoder[i] for i in top_indices]
            recommendations.append(top_places)

        return pd.DataFrame({"user_id": model_input["user_id"], "recommendations": recommendations})
