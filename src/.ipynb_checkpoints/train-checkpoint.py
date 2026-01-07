# src/train.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import sys
from pathlib import Path
import json

# Ajouter la racine du projet au path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from .load_save_data import upload_to_gcs, save_local, load_local_all_data, load_data_from_gcs, save_json_to_both
import pickle

def train_best_model(train_df, best_params, save_mode="cloud"):
    """Entraîne et sauvegarde le modèle."""
    from sklearn.preprocessing import LabelEncoder

    # Encodeurs
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    user_encoder.fit(train_df['userId'].unique())
    movie_encoder.fit(train_df['movieId'].unique())
    print(f"Encoded {len(user_encoder.classes_)} users and {len(movie_encoder.classes_)} movies")

    # Colonnes indices
    if 'user_idx' not in train_df.columns or 'movie_idx' not in train_df.columns:
        train_df['user_idx'] = user_encoder.transform(train_df['userId'])
        train_df['movie_idx'] = movie_encoder.transform(train_df['movieId'])

    # Dataset surprise
    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    data = Dataset.load_from_df(train_df[['user_idx', 'movie_idx', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Modèle SVD avec hyperparamètres
    algo = SVD(**best_params)
    algo.fit(trainset)

    # Sauvegarde
    if save_mode == "cloud":
        upload_to_gcs(pickle.dumps(algo), "svd_model.pkl")
        upload_to_gcs(pickle.dumps(user_encoder), "user_encoder.pkl")
        upload_to_gcs(pickle.dumps(movie_encoder), "movie_encoder.pkl")
    else:
        save_local(algo, "svd_model.pkl")
        save_local(user_encoder, "user_encoder.pkl")
        save_local(movie_encoder, "movie_encoder.pkl")

    return algo, user_encoder, movie_encoder

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cloud"

    # 1️⃣ Charger données
    if mode == "cloud":
        _ , _ , df_ratings = load_data_from_gcs()
    else:
        _ , _ , df_ratings = load_local_all_data()

    # 2️⃣ GridSearch SVD
    reader = Reader(rating_scale=(df_ratings['rating'].min(), df_ratings['rating'].max()))
    data = Dataset.load_from_df(df_ratings[['userId','movieId','rating']], reader)
    gs = GridSearchCV(SVD,
                      param_grid={
                          'n_factors':[20,50,100],
                          'n_epochs':[20,30],
                          'lr_all':[0.002,0.005],
                          'reg_all':[0.02,0.05]
                      },
                      measures=['rmse'], cv=3, n_jobs=-1)
    print("🔍 Recherche des meilleurs hyperparamètres SVD...")
    gs.fit(data)
    best_params = gs.best_params['rmse']
    save_json_to_both(best_params, "svd_best_params.json")
    print("✅ Meilleurs paramètres sauvegardés.")

    # 3️⃣ Entraîner et sauvegarder le modèle final
    train_best_model(df_ratings, best_params, save_mode=mode)

