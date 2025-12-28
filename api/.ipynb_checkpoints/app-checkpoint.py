from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# Définir les chemins relatifs
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "../models")
DATA_DIR = os.path.join(BASE_DIR, "../data")

# Charger modèles et encoders
with open(os.path.join(MODELS_DIR, "knn_model.pkl"), "rb") as f:
    algo = pickle.load(f)
with open(os.path.join(MODELS_DIR, "user_encoder.pkl"), "rb") as f:
    user_encoder = pickle.load(f)
with open(os.path.join(MODELS_DIR, "movie_encoder.pkl"), "rb") as f:
    movie_encoder = pickle.load(f)

df_movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

# Créer l'API
app = FastAPI(title="Movie Recommender API")

# Modèle de requête
class RatingRequest(BaseModel):
    user_id: int
    movie_ids: list[int] | None = None  # facultatif, si pas fourni -> top-N

TOP_N = 5

@app.post("/predict")
def predict_ratings(req: RatingRequest):
    # Transformer user_id en user_idx
    try:
        user_idx = user_encoder.transform([req.user_id])[0]
    except ValueError:
        raise HTTPException(status_code=404, detail=f"User {req.user_id} not found")

    # Si une liste de films est fournie
    if req.movie_ids:
        results = []
        for movie_id in req.movie_ids:
            try:
                movie_idx = movie_encoder.transform([movie_id])[0]
            except ValueError:
                continue  # ignorer les films inconnus

            pred_rating = algo.predict(user_idx, movie_idx).est
            title = df_movies.loc[df_movies['movieId']==movie_id, 'title'].values
            title_str = title[0] if len(title) > 0 else "Unknown Title"
            results.append({
                "movie_id": movie_id,
                "title": title_str,
                "predicted_rating": round(pred_rating, 2)
            })
        return {"user_id": req.user_id, "predictions": results}

    # Sinon, top-N recommandations
    all_movie_idx = list(movie_encoder.transform(movie_encoder.classes_))
    pred_for_user = []
    for movie_idx in all_movie_idx:
        pred_rating = algo.predict(user_idx, movie_idx).est
        movie_id = movie_encoder.inverse_transform([movie_idx])[0]
        title = df_movies.loc[df_movies['movieId']==movie_id, 'title'].values
        title_str = title[0] if len(title) > 0 else "Unknown Title"
        pred_for_user.append({
            "movie_id": movie_id,
            "title": title_str,
            "predicted_rating": round(pred_rating, 2)
        })

    top_recommendations = sorted(pred_for_user, key=lambda x: x["predicted_rating"], reverse=True)[:TOP_N]
    return {"user_id": req.user_id, "top_recommendations": top_recommendations}
