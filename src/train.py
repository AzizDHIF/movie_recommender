# src/train.py

import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_best_model(train_df, models_dir="../models"):
    
    """
    Entraîne le meilleur modèle (SVD) sur le train_df et sauvegarde le modèle.
    
    Args:
        train_df (pd.DataFrame): DataFrame avec colonnes ['user_idx', 'movie_idx', 'rating']
        models_dir (str): chemin où sauvegarder le modèle
    
    Returns:
        algo (SVD): modèle SVD entraîné
    """

    # 1. Préparer le dataset pour surprise
    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    data = Dataset.load_from_df(train_df[['user_idx', 'movie_idx', 'rating']], reader)
    trainset = data.build_full_trainset()  # Utiliser tout le train_df

    # 2. Créer et entraîner le modèle SVD
    algo = SVD(random_state=42)
    print("Entraînement du modèle SVD...")
    algo.fit(trainset)
    print("Entraînement terminé !")

    # 3. Créer le dossier models s'il n'existe pas
    os.makedirs(models_dir, exist_ok=True)

    # 4. Sauvegarder le modèle
    model_path = os.path.join(models_dir, "svd_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(algo, f)
    print(f"Modèle SVD sauvegardé dans {model_path}")

    return algo


if __name__ == "__main__":
    # Exemple d'utilisation directe
    # train_df doit être un fichier csv ou pickle contenant ['user_idx', 'movie_idx', 'rating']
    train_df = pd.read_csv("../data/train_df.csv")  # adapter le chemin selon ton projet
    train_best_model(train_df)
