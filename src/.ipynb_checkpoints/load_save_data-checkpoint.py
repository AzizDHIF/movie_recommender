
import pickle
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVD
from google.cloud import storage

BUCKET_NAME = "movie-reco-models-fatma-aziz-students-group2"

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

def load_local_data():
    """Charge les données locales"""
    data_path = Path(__file__).parent.parent / "data" / "train_ratings.csv"
    return pd.read_csv(data_path)

def load_gcs_data():
    """Charge depuis GCS"""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob("data/train_ratings.csv")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))