from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd

# Importer fonctions depuis src/recommend.py
from src.recommend import load_model_and_encoders, recommend_movies

# Définir les chemins relatifs
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "../models")
DATA_DIR = os.path.join(BASE_DIR, "../data")

# Charger le modèle et les encoders dynamiquement
algo, user_encoder, movie_encoder = load_model_and_encoders(models_dir=MODELS_DIR)

# Charger la table des films
df_movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

# Créer l'API FastAPI
app = FastAPI(title="Movie Recommender API")

# Modèle de requête
class RatingRequest(BaseModel):
    user_id: int
    movie_ids: list[int] | None = None  # facultatif, si non fourni -> top-N

TOP_N = 5

@app.post("/predict")
def predict_ratings(req: RatingRequest):
    import pandas as pd
    # Charger le dataframe d'entraînement pour connaître les utilisateurs existants
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_df.csv"))

    # Si des films spécifiques sont fournis
    if req.movie_ids:
        results = []
        for movie_id in req.movie_ids:
            try:
                user_idx = user_encoder.transform([req.user_id])[0]
            except ValueError:
                # Nouveau utilisateur -> créer un index fictif
                user_idx = max(train_df['user_idx']) + 1

            try:
                movie_idx = movie_encoder.transform([movie_id])[0]
            except ValueError:
                continue  # ignorer films inconnus

            pred_rating = algo.predict(user_idx, movie_idx).est
            title = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values
            title_str = title[0] if len(title) > 0 else "Unknown Title"
            results.append({
                "movie_id": movie_id,
                "title": title_str,
                "predicted_rating": round(pred_rating, 2)
            })
        return {"user_id": req.user_id, "predictions": results}

    # Sinon, top-N recommandations
    recommendations = recommend_movies(
        user_id=req.user_id,
        train_df=train_df,
        df_movies=df_movies,
        algo=algo,
        user_encoder=user_encoder,
        movie_encoder=movie_encoder,
        top_n=TOP_N
    )

    return {
        "user_id": req.user_id,
        "top_recommendations": [
            {"title": title, "predicted_rating": round(rating, 2)}
            for title, rating in recommendations
        ]
    }

# Endpoint pour réentraînement futur
@app.post("/retrain")
def retrain_model():
    from src.train import train_best_model
    train_best_model()  # fonction qui entraîne SVD et met à jour les pickle
    # Recharger le modèle après réentraînement
    global algo, user_encoder, movie_encoder
    algo, user_encoder, movie_encoder = load_model_and_encoders(models_dir=MODELS_DIR)
    return {"message": "Modèle réentraîné et mis à jour"}
