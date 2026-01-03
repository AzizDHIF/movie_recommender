# src/train.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from .load_save_data import upload_to_gcs, save_local, load_local_data, load_data_from_gcs

# Dans src/train.py, modifiez la création des encodeurs :

def train_best_model(train_df, save_mode="cloud"):
    """Entraîne et sauvegarde le modèle."""
    from sklearn.preprocessing import LabelEncoder
    import pickle
    
    # 1. Créer les encodeurs LabelEncoder
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    # 2. Fit avec les IDs uniques
    user_encoder.fit(train_df['userId'].unique())
    movie_encoder.fit(train_df['movieId'].unique())
    
    print(f"Encoded {len(user_encoder.classes_)} users and {len(movie_encoder.classes_)} movies")
    
    # 3. S'assurer que les colonnes user_idx et movie_idx existent
    if 'user_idx' not in train_df.columns or 'movie_idx' not in train_df.columns:
        print("Adding idx columns...")
        train_df = train_df.copy()
        train_df['user_idx'] = user_encoder.transform(train_df['userId'])
        train_df['movie_idx'] = movie_encoder.transform(train_df['movieId'])
    
    # 4. Entraînement (inchangé)
    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    data = Dataset.load_from_df(train_df[['user_idx', 'movie_idx', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    algo = SVD(random_state=42)
    algo.fit(trainset)
    
    # 5. Sauvegarde
    if save_mode == "cloud":
        # Sauvegarder comme pickle
        upload_to_gcs(pickle.dumps(algo), "svd_model.pkl")
        upload_to_gcs(pickle.dumps(user_encoder), "user_encoder.pkl") 
        upload_to_gcs(pickle.dumps(movie_encoder), "movie_encoder.pkl")
    elif save_mode == "local":
        save_local(algo, "svd_model.pkl")
        save_local(user_encoder, "user_encoder.pkl")
        save_local(movie_encoder, "movie_encoder.pkl")
    
    return algo, user_encoder, movie_encoder


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cloud"
    
    # Chargement
    if mode == "cloud":
        _ , _ , df_ratings = load_data_from_gcs()
    else:
        _ , _ , df_ratings = load_local_all_data()
    
    # Entraînement
    train_best_model(df_ratings, mode)