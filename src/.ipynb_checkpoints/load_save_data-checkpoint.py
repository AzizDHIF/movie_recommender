import pickle
import pandas as pd
from pathlib import Path
from google.cloud import storage
import io
import numpy as np
import json

BUCKET_NAME = "movie-reco-models-fatma-aziz-students-group2"

# ==================== SAUVEGARDE ====================

def upload_to_gcs(obj, blob_name):
    """Sauvegarde sur GCS"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{blob_name}")
    blob.upload_from_string(pickle.dumps(obj))
    print(f"✅ {blob_name} → GCS")

def save_local(obj, filename):
    """Sauvegarde locale"""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ {filename} → local")

def save_to_both(obj, filename):
    """Sauvegarde à la fois local et cloud"""
    save_local(obj, filename)
    upload_to_gcs(obj, filename)

# ==================== CHARGEMENT DONNÉES ====================

def load_local_data():
    """Charge les données locales d'entraînement"""
    data_path = Path(__file__).parent.parent / "data" / "train_ratings.csv"
    return pd.read_csv(data_path)

def load_data_from_gcs():
    """Charge les données depuis GCS"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    
    df_blob = bucket.blob("data/df_ratings.csv")
    train_blob = bucket.blob("data/train_ratings.csv")
    movies_blob = bucket.blob("data/movies.csv")

    train_df = pd.read_csv(io.BytesIO(train_blob.download_as_bytes()))
    df_movies = pd.read_csv(io.BytesIO(movies_blob.download_as_bytes()))
    df_ratings = pd.read_csv(io.BytesIO(df_blob.download_as_bytes()))
    return train_df, df_movies, df_ratings

def load_local_all_data():
    """Charge toutes les données locales"""
    data_dir = Path(__file__).parent.parent / "data"
    train_df = pd.read_csv(data_dir / "train_ratings.csv")
    df_movies = pd.read_csv(data_dir / "movies.csv")
    df_ratings = pd.read_csv(data_dir / "df_ratings.csv")
    return train_df, df_movies,df_ratings

# ==================== CHARGEMENT MODÈLES ====================

def load_model_and_encoders_from_gcs():
    """Charge modèle et encoders depuis GCS"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    def load_pickle_from_gcs(blob_name):
        blob = bucket.blob(f"models/{blob_name}")
        data = blob.download_as_bytes()
        return pickle.loads(data)

    algo = load_pickle_from_gcs("svd_model.pkl")
    user_encoder = load_pickle_from_gcs("user_encoder.pkl")
    movie_encoder = load_pickle_from_gcs("movie_encoder.pkl")
    
     # Vérifier le type des encodeurs
    print(f"📦 Type user_encoder: {type(user_encoder)}")
    print(f"📦 Type movie_encoder: {type(movie_encoder)}")

    return algo, user_encoder, movie_encoder

def load_model_and_encoders_local():
    """Charge modèle et encoders depuis local"""
    models_dir = Path(__file__).parent.parent / "models"
    
    def load_pickle_local(blob_name):
        with open(models_dir / blob_name, 'rb') as f:
            return pickle.load(f)

    algo = load_pickle_local("svd_model.pkl")
    user_encoder = load_pickle_local("user_encoder.pkl")
    movie_encoder = load_pickle_local("movie_encoder.pkl")
    
     # Vérifier le type des encodeurs
    print(f"📦 Type user_encoder: {type(user_encoder)}")
    print(f"📦 Type movie_encoder: {type(movie_encoder)}")

    return algo, user_encoder, movie_encoder

# ==================== UTILITAIRES ====================

def load_data(source="local"):
    """Charge les données depuis la source spécifiée"""
    if source == "local":
        return load_local_all_data()
    elif source == "cloud":
        return load_data_from_gcs()
    else:
        raise ValueError("Source doit être 'local' ou 'cloud'")

def load_model(source="local"):
    """Charge le modèle depuis la source spécifiée"""
    if source == "local":
        return load_model_and_encoders_local()
    elif source == "cloud":
        return load_model_and_encoders_from_gcs()
    else:
        raise ValueError("Source doit être 'local' ou 'cloud'")
        
        

# ==================== SAUVEGARDE DATA ====================

def save_data_local(df, filename):
    """Sauvegarde CSV en local"""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / filename, index=False)
    print(f"✅ {filename} → local")

def upload_csv_to_gcs(df, filename):
    """Sauvegarde CSV sur GCS"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(f"data/{filename}")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")

    print(f"✅ {filename} → GCS")

def save_data_to_both(df, filename):
    """Sauvegarde data en local + cloud"""
    save_data_local(df, filename)
    upload_csv_to_gcs(df, filename)
    
    
def save_json_local(data: dict, filename: str):
    """Sauvegarde JSON en local"""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"✅ {filename} → local (JSON)")

def upload_json_to_gcs(data: dict, filename: str):
    """Sauvegarde JSON sur GCS"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{filename}")

    blob.upload_from_string(
        json.dumps(data, indent=4),
        content_type="application/json"
    )

    print(f"✅ {filename} → GCS (JSON)")

    
def save_json_to_both(data: dict, filename: str):
    """Sauvegarde JSON en local + cloud"""
    save_json_local(data, filename)
    upload_json_to_gcs(data, filename)

