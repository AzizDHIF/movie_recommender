from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Charger modèles
models_dir = "models"
with open(os.path.join(models_dir, "knn_model.pkl"), "rb") as f:
    algo = pickle.load(f)
with open(os.path.join(models_dir, "user_encoder.pkl"), "rb") as f:
    user_encoder = pickle.load(f)
with open(os.path.join(models_dir, "movie_encoder.pkl"), "rb") as f:
    movie_encoder = pickle.load(f)

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 5):
    # TODO : convertir user_id -> user_idx, filtrer films vus
    # TODO : calculer top-N recommandations
    # Retourner un JSON avec movieId, title, predicted_rating
    return {"user_id": user_id, "recommendations": []}
