# src/recommend.py

import os
import pickle
import pandas as pd
from surprise import SVD

def load_model_and_encoders(models_dir="../models"):
    """
    Charge le modèle SVD et les encoders utilisateur/film.
    """
    # Charger le modèle SVD
    model_path = os.path.join(models_dir, "svd_model.pkl")
    with open(model_path, "rb") as f:
        algo = pickle.load(f)

    # Charger les encoders
    with open(os.path.join(models_dir, "user_encoder.pkl"), "rb") as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(models_dir, "movie_encoder.pkl"), "rb") as f:
        movie_encoder = pickle.load(f)

    return algo, user_encoder, movie_encoder


def recommend_movies(user_id, train_df, df_movies, algo, user_encoder, movie_encoder, top_n=5):
    """
    Génère les top N recommandations pour un utilisateur.

    Args:
        user_id (int): ID utilisateur original
        train_df (pd.DataFrame): dataframe d'entraînement
        df_movies (pd.DataFrame): dataframe des films avec 'movieId' et 'title'
        algo (SVD): modèle SVD entraîné
        user_encoder: encoder des utilisateurs
        movie_encoder: encoder des films
        top_n (int): nombre de recommandations

    Returns:
        list of tuples: [(titre_film, note_predite), ...]
    """
    # Encoder l'utilisateur
    if user_id in train_df['userId'].values:
        user_idx = user_encoder.transform([user_id])[0]
    else:
        # si nouvel utilisateur, assigner un index fictif
        user_idx = max(train_df['user_idx']) + 1
        print(f"Nouvel utilisateur détecté. Assigné un user_idx = {user_idx}")

    # Films déjà vus
    movies_watched = train_df[train_df['userId'] == user_id]['movie_idx'].tolist()

    # Tous les films connus
    all_movie_idx = list(range(len(movie_encoder.classes_)))

    # Prédictions pour les films non vus
    pred_for_user = []
    for movie_idx in all_movie_idx:
        if movie_idx not in movies_watched:
            pred_rating = algo.predict(user_idx, movie_idx).est
            pred_for_user.append((movie_idx, pred_rating))

    # Trier par note prédite décroissante
    top_recommendations = sorted(pred_for_user, key=lambda x: x[1], reverse=True)[:top_n]

    # Convertir movie_idx en titre
    recommendations = []
    for movie_idx, rating in top_recommendations:
        movie_id = movie_encoder.inverse_transform([movie_idx])[0]
        title = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values[0]
        recommendations.append((title, rating))

    return recommendations


if __name__ == "__main__":
    # Exemple d'utilisation
    train_df = pd.read_csv("../data/train_df.csv")  # adapter le chemin
    df_movies = pd.read_csv("../data/movies.csv")   # adapter le chemin

    algo, user_encoder, movie_encoder = load_model_and_encoders()

    user_id = 50  # exemple
    recommendations = recommend_movies(user_id, train_df, df_movies, algo, user_encoder, movie_encoder, top_n=5)

    print(f"\nTop 5 recommandations pour l'utilisateur {user_id} :\n")
    for title, rating in recommendations:
        print(f"{title} - predicted rating: {rating:.2f}")
