from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# Importer fonctions depuis src/recommend.py et src/train.py
from src.recommend import (
    load_model_and_encoders_from_gcs,
    load_data_from_gcs,
    recommend_movies
)
from src.train import train_best_model

# Charger modèle, encoders et données depuis GCS
algo, user_encoder, movie_encoder = load_model_and_encoders_from_gcs()
train_df, df_movies = load_data_from_gcs()

# Créer l'API
app = FastAPI(title="Movie Recommender API")

# Modèle de requête
class RatingRequest(BaseModel):
    user_id: int
    movie_ids: list[int] | None = None  # facultatif, si pas fourni -> top-N

TOP_N = 5


@app.post("/predict")
def predict_ratings(req: RatingRequest):
    global train_df

    # =========================
    # 1️⃣ Vérifier l'utilisateur
    # =========================
    try:
        user_idx = user_encoder.transform([req.user_id])[0]
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {req.user_id} inconnu. "
                   "Le modèle doit être réentraîné pour supporter de nouveaux utilisateurs."
        )

    results = []

    # =========================
    # 2️⃣ Si movie_ids fournis
    # =========================
    if req.movie_ids:
        for movie_id in req.movie_ids:
            # Vérifier que le film existe dans le modèle
            try:
                movie_idx = movie_encoder.transform([movie_id])[0]
            except ValueError:
                continue  # ignorer les films inconnus

            # Ajouter une note simulée (ex: 4.0)
            new_row = {
                "userId": req.user_id,
                "movieId": movie_id,
                "user_idx": user_idx,
                "movie_idx": movie_idx,
                "rating": 4.0
            }

            train_df = pd.concat(
                [train_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            pred_rating = algo.predict(user_idx, movie_idx).est
            title = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values
            title_str = title[0] if len(title) > 0 else "Unknown Title"
            results.append({
                "movie_id": movie_id,
                "title": title_str,
                "predicted_rating": round(pred_rating, 2)
            })

        return {"user_id": req.user_id, "predictions": results}

    # =========================
    # 3️⃣ Sinon, top-N recommandations
    # =========================
    
    # LIGNE 1 MODIFIÉE : Récupérer les films notés APRÈS les mises à jour
    user_rated_movies = train_df[train_df['userId'] == req.user_id]['movieId'].unique().tolist()
    
    all_movie_idx = list(movie_encoder.transform(movie_encoder.classes_))
    pred_for_user = []
    
    for movie_idx in all_movie_idx:
        movie_id = movie_encoder.inverse_transform([movie_idx])[0]
        
        # LIGNE 2 MODIFIÉE : Utiliser la liste complète (unique)
        if movie_id in user_rated_movies:
            continue
            
        pred_rating = algo.predict(user_idx, movie_idx).est
        title = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values
        title_str = title[0] if len(title) > 0 else "Unknown Title"
        pred_for_user.append({
            "movie_id": movie_id,
            "title": title_str,
            "predicted_rating": round(pred_rating, 2)
        })

    top_recommendations = sorted(pred_for_user, key=lambda x: x["predicted_rating"], reverse=True)[:TOP_N]

    return {"user_id": req.user_id, "top_recommendations": top_recommendations}

@app.post("/retrain")
def retrain_model():
    global algo, user_encoder, movie_encoder, train_df

    # Réentraînement du modèle sur train_df et sauvegarde sur GCS
    algo, user_encoder, movie_encoder = train_best_model(train_df)

    return {"message": "Modèle réentraîné et mis à jour sur le cloud"}
