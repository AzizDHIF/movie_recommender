# src/train.py

import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from google.cloud import storage
import io

# Nom de ton bucket GCS
BUCKET_NAME = "movie-reco-models-fatma-aziz-students-group2"

def upload_to_gcs(obj, blob_name, bucket_name=BUCKET_NAME):
    """
    Sauvegarde un objet Python pickle sur GCS.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(f"models/{blob_name}")
    blob.upload_from_string(pickle.dumps(obj))
    print(f"{blob_name} uploaded to {bucket_name} ✅")

def train_best_model(train_df):
    """
    Entraîne le meilleur modèle (SVD) sur train_df et sauvegarde modèle + encoders sur GCS.
    
    Args:
        train_df (pd.DataFrame): colonnes ['user_idx', 'movie_idx', 'rating']
    
    Returns:
        algo (SVD): modèle SVD entraîné
    """

    # 1. Préparer le dataset pour Surprise
    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    data = Dataset.load_from_df(train_df[['user_idx', 'movie_idx', 'rating']], reader)
    trainset = data.build_full_trainset()  # Utiliser tout le train_df

    # 2. Créer et entraîner le modèle SVD
    algo = SVD(random_state=42)
    print("Entraînement du modèle SVD...")
    algo.fit(trainset)
    print("Entraînement terminé !")

    # 3. Créer les encoders (user et movie)
    user_encoder = {uid: idx for idx, uid in enumerate(train_df['userId'].unique())}
    movie_encoder = {mid: idx for idx, mid in enumerate(train_df['movieId'].unique())}

    # 4. Sauvegarder le modèle et les encoders sur GCS
    upload_to_gcs(algo, "svd_model.pkl")
    upload_to_gcs(user_encoder, "user_encoder.pkl")
    upload_to_gcs(movie_encoder, "movie_encoder.pkl")

    return algo, user_encoder, movie_encoder


if __name__ == "__main__":
    # Lire le fichier train_ratings.csv depuis le dossier data
    train_df = pd.read_csv("../data/train_ratings.csv")

    # Entraîner et sauvegarder sur le cloud
    algo, user_encoder, movie_encoder = train_best_model(train_df)

