# src/train.py
import pickle
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVD
from google.cloud import storage
from load_save_data import upload_to_gcp,save_local,load_data_from_gcs,load_local_data
 
def train_best_model(train_df, save_mode="cloud"):
    """
    Entraîne le modèle SVD.
    
    Args:
        train_df: Données d'entraînement
        save_mode: "cloud" (GCS) ou "local" (fichiers locaux)
    """
    # Entraînement
    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    data = Dataset.load_from_df(train_df[['user_idx', 'movie_idx', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    algo = SVD(random_state=42)
    algo.fit(trainset)
    
    # Créer encoders
    user_encoder = {uid: idx for idx, uid in enumerate(train_df['userId'].unique())}
    movie_encoder = {mid: idx for idx, mid in enumerate(train_df['movieId'].unique())}
    
    # Sauvegarde selon le mode
    if save_mode == "cloud":
        upload_to_gcs(algo, "svd_model.pkl")
        upload_to_gcs(user_encoder, "user_encoder.pkl")
        upload_to_gcs(movie_encoder, "movie_encoder.pkl")
    elif save_mode == "local":
        save_local(algo, "svd_model.pkl")
        save_local(user_encoder, "user_encoder.pkl")
        save_local(movie_encoder, "movie_encoder.pkl")
    
    return algo, user_encoder, movie_encoder

if __name__ == "__main__":
    # Usage: python train.py cloud   OU   python train.py local
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cloud"
    if mode == "cloud":
        train_df,x = load_data_from_gcs()
    elif mode == "local":
        train_df = load_local_data()
        
    # train the model 
    train_best_model(train_df, mode)