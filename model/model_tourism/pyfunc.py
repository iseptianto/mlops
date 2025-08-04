import mlflow.pyfunc
import pickle
import numpy as np

class TourismRecommenderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load semua file pkl saat model diload
        self.user_encoder = pickle.load(open(context.artifacts["user_encoder"], "rb"))
        self.place_encoder = pickle.load(open(context.artifacts["place_encoder"], "rb"))
        self.prediction_matrix = pickle.load(open(context.artifacts["prediction_matrix"], "rb"))

    def predict(self, context, model_input):
        """
        model_input: DataFrame with column 'user_id'
        Returns: List of recommendations per user
        """
        rekomendasi_all = []

        for user in model_input["user_id"]:
            if user not in self.user_encoder:
                rekomendasi_all.append(["Unknown User"])
                continue

            # Index user di prediction_matrix
            user_idx = self.user_encoder[user]

            # Skor preferensi tempat wisata
            user_scores = np.array(self.prediction_matrix[user_idx])

            # Urutan skor tertinggi
            top_indices = user_scores.argsort()[-3:][::-1]

            # Nama tempat wisata
            top_places = [self.place_encoder[i] for i in top_indices]

            rekomendasi_all.append(top_places)

        return rekomendasi_all
