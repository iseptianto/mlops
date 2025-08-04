import pickle
import numpy as np

# user_encoder.pkl
user_encoder = {"userA": 0, "userB": 1, "userC": 2}
pickle.dump(user_encoder, open("user_encoder.pkl", "wb"))

# place_encoder.pkl
place_encoder = {0: "Borobudur", 1: "Prambanan", 2: "Raja Ampat"}
pickle.dump(place_encoder, open("place_encoder.pkl", "wb"))

# prediction_matrix.pkl
prediction_matrix = np.array([
    [0.7, 0.9, 0.2],  # userA
    [0.1, 0.5, 0.8],  # userB
    [0.3, 0.4, 0.1],  # userC
])
pickle.dump(prediction_matrix, open("prediction_matrix.pkl", "wb"))
