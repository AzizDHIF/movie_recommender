import pickle
import pandas as pd
from surprise import SVD

# Import relatif (même dossier)
from .load_save_data import (
    load_model_and_encoders_from_gcs, 
    load_model_and_encoders_local,
    load_data_from_gcs,
    load_local_all_data,
    load_model,
    load_data
)

def recommend_movies(user_id, train_df, df_movies, algo, user_encoder, movie_encoder, top_n=5):
    """
    Génère les top N recommandations pour un utilisateur.
    """
    # Encoder l'utilisateur
    if user_id in train_df['userId'].values:
        user_idx = user_encoder[user_id]
    else:
        user_idx = max(train_df['user_idx']) + 1
        print(f"Nouvel utilisateur détecté. Assigné user_idx = {user_idx}")

    # Films déjà vus
    movies_watched = train_df[train_df['userId'] == user_id]['movie_idx'].tolist()

    # Tous les films
    all_movie_idx = list(range(len(movie_encoder)))

    # Prédictions
    pred_for_user = []
    for movie_idx in all_movie_idx:
        if movie_idx not in movies_watched:
            pred_rating = algo.predict(user_idx, movie_idx).est
            pred_for_user.append((movie_idx, pred_rating))

    # Trier et limiter
    top_recommendations = sorted(pred_for_user, key=lambda x: x[1], reverse=True)[:top_n]

    # Convertir en titres
    recommendations = []
    for movie_idx, rating in top_recommendations:
        # Trouver le movie_id correspondant
        movie_id = None
        for mid, idx in movie_encoder.items():
            if idx == movie_idx:
                movie_id = mid
                break
        
        if movie_id:
            title = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values[0]
            recommendations.append((movie_id, title, rating))

    return recommendations

def get_recommendations(user_id, source="cloud", top_n=5):
    """Fonction principale avec choix de source"""
    if source == "cloud":
        algo, user_encoder, movie_encoder = load_model_and_encoders_from_gcs()
        train_df, df_movies = load_data_from_gcs()
    elif source == "local":
        algo, user_encoder, movie_encoder = load_model_and_encoders_local()
        train_df, df_movies = load_local_all_data()
    else:
        raise ValueError("Source doit être 'cloud' ou 'local'")
    
    return recommend_movies(user_id, train_df, df_movies, algo, user_encoder, movie_encoder, top_n)

if __name__ == "__main__":
    import sys
    
    # Interface CLI
    if len(sys.argv) > 1:
        user_id = int(sys.argv[1])
        source = sys.argv[2] if len(sys.argv) > 2 else "cloud"
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    else:
        user_id = 50
        source = "cloud"
        top_n = 5
    
    recommendations = get_recommendations(user_id, source, top_n)
    
    print(f"\nTop {len(recommendations)} recommandations pour l'utilisateur {user_id} :\n")
    for movie_id, title, rating in recommendations:
        print(f"{title} - {rating:.2f} (ID: {movie_id})")