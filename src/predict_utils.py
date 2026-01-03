"""
Utils pour les prédictions avec SVD.
"""
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import LabelEncoder
import pickle

def load_and_prepare_model(algo, train_df, user_encoder, movie_encoder):
    """
    Prépare le modèle pour les prédictions.
    """
    # Créer un trainset
    surprise_data = train_df[['userId', 'movieId', 'rating']].copy()
    surprise_data['userId'] = surprise_data['userId'].astype(str)
    surprise_data['movieId'] = surprise_data['movieId'].astype(str)
    surprise_data['rating'] = surprise_data['rating'].astype(float)
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(surprise_data, reader)
    trainset = data.build_full_trainset()
    
    # Ré-entraîner le modèle
    algo.fit(trainset)
    
    return algo, trainset

def predict_for_user(user_id, algo, trainset, df_movies, n_recommendations=5):
    """
    Prédictions correctes pour un utilisateur.
    """
    # Convertir user_id en ID interne de surprise
    user_inner = trainset.to_inner_uid(str(user_id))
    
    # Obtenir tous les films
    all_movies = []
    for movie_inner in range(trainset.n_items):
        try:
            # Obtenir l'ID original du film
            movie_raw = trainset.to_raw_iid(movie_inner)
            movie_id = int(movie_raw)
            
            # Prédire
            pred = algo.predict(user_inner, movie_inner)
            
            # Obtenir le titre
            title_row = df_movies[df_movies['movieId'] == movie_id]
            if len(title_row) > 0:
                title = title_row['title'].iloc[0]
            else:
                title = f"Movie {movie_id}"
            
            all_movies.append({
                'movie_id': movie_id,
                'title': title,
                'predicted_rating': pred.est
            })
        except:
            continue
    
    # Trier et retourner les meilleurs
    all_movies.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return all_movies[:n_recommendations]

# Version simplifiée pour votre API
def get_recommendations_fixed(user_id, train_df, df_movies, algo, user_encoder, movie_encoder):
    """
    Fonction fixée pour l'API.
    """
    # Préparer le modèle
    algo_prepared, trainset = load_and_prepare_model(algo, train_df, user_encoder, movie_encoder)
    
    # Obtenir les recommandations
    recommendations = predict_for_user(user_id, algo_prepared, trainset, df_movies)
    
    return recommendations