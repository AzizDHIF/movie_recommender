# === AJOUTER AU DÉBUT DE api/app.py ===
from surprise import Dataset, Reader
from .load_save_data import *

def create_surprise_trainset(train_df):
    """Crée un trainset pour le modèle surprise"""
    # Préparer les données
    surprise_data = train_df[['userId', 'movieId', 'rating']].copy()
    surprise_data['userId'] = surprise_data['userId'].astype(str)
    surprise_data['movieId'] = surprise_data['movieId'].astype(str)
    surprise_data['rating'] = surprise_data['rating'].astype(float)
    
    # Créer trainset
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(surprise_data, reader)
    trainset = data.build_full_trainset()
    
    return trainset

def ensure_algo_has_trainset(algo, train_df):
    """S'assure que l'algorithme a un trainset"""
    if not hasattr(algo, 'trainset') or algo.trainset is None:
        # Créer un trainset et ré-entraîner
        trainset = create_surprise_trainset(train_df)
        algo.fit(trainset)
    return algo

def quick_compare_cloud_vs_local():
    """
    Comparaison rapide cloud vs local
    Retourne True si tout est identique, False sinon
    """
    try:
        # Données
        train_gcs, movies_gcs, ratings_gcs = load_data_from_gcs()
        train_local, movies_local, ratings_local = load_local_all_data()
        
        # Vérification des dimensions
        if not (train_gcs.shape == train_local.shape and
                movies_gcs.shape == movies_local.shape and
                ratings_gcs.shape == ratings_local.shape):
            print("⚠️ Dimensions différentes")
            return False
        
        # Vérification des colonnes
        datasets_gcs = [train_gcs, movies_gcs, ratings_gcs]
        datasets_local = [train_local, movies_local, ratings_local]
        names = ["train_ratings", "movies", "df_ratings"]
        
        for name, df_gcs, df_local in zip(names, datasets_gcs, datasets_local):
            if not set(df_gcs.columns) == set(df_local.columns):
                print(f"⚠️ Colonnes différentes pour {name}")
                return False
        
        # Vérification des valeurs (échantillon)
        for name, df_gcs, df_local in zip(names, datasets_gcs, datasets_local):
            # Comparer les 100 premières lignes
            sample_gcs = df_gcs.head(100)
            sample_local = df_local.head(100)
            
            if not sample_gcs.equals(sample_local):
                print(f"⚠️ Valeurs différentes pour {name}")
                return False
        
        # Modèles
        algo_gcs, user_enc_gcs, movie_enc_gcs = load_model_and_encoders_from_gcs()
        algo_local, user_enc_local, movie_enc_local = load_model_and_encoders_local()
        
        # Vérifier les classes des encodeurs
        if (not np.array_equal(user_enc_gcs.classes_, user_enc_local.classes_) or
            not np.array_equal(movie_enc_gcs.classes_, movie_enc_local.classes_)):
            print("⚠️ Encodeurs différents")
            return False
        
        print("✅ Toutes les données et modèles sont synchronisés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")
        return False
    

def check_model_data_compatibility(algo, user_encoder, movie_encoder, train_df):
    """
    Vérifie la compatibilité entre le modèle et les données
    """
    print("🔍 Vérification de compatibilité Modèle-Données")
    
    # 1. Vérifier les colonnes nécessaires
    required_cols = {'user_idx', 'movie_idx'}
    available_cols = set(train_df.columns)
    
    if not required_cols.issubset(available_cols):
        print(f"❌ Colonnes manquantes dans train_df: {required_cols - available_cols}")
        return False
    
    print(f"✅ Colonnes nécessaires présentes: {required_cols}")
    
    # 2. Vérifier les plages d'indices
    unique_user_ids = train_df['user_idx'].unique()
    unique_movie_ids = train_df['movie_idx'].unique()
    
    print(f"📊 Plage user_idx: {unique_user_ids.min()} - {unique_user_ids.max()} (total: {len(unique_user_ids)})")
    print(f"📊 Plage movie_idx: {unique_movie_ids.min()} - {unique_movie_ids.max()} (total: {len(unique_movie_ids)})")
    
    # 3. Vérifier avec les encodeurs
    if hasattr(user_encoder, 'classes_'):
        encoder_user_range = len(user_encoder.classes_)
        data_user_range = unique_user_ids.max() + 1  # indices commencent à 0
        
        print(f"👤 User encoder: {encoder_user_range} classes")
        print(f"👤 Data user_idx max: {data_user_range}")
        
        if data_user_range > encoder_user_range:
            print(f"⚠️ Attention: user_idx dans données ({data_user_range}) > encoder ({encoder_user_range})")
            return False
    
    if hasattr(movie_encoder, 'classes_'):
        encoder_movie_range = len(movie_encoder.classes_)
        data_movie_range = unique_movie_ids.max() + 1
        
        print(f"🎬 Movie encoder: {encoder_movie_range} classes")
        print(f"🎬 Data movie_idx max: {data_movie_range}")
        
        if data_movie_range > encoder_movie_range:
            print(f"⚠️ Attention: movie_idx dans données ({data_movie_range}) > encoder ({encoder_movie_range})")
            return False
    
    # 4. Tester une prédiction simple - CORRECTION ICI
    try:
        test_user = unique_user_ids[0]
        test_movie = unique_movie_ids[0]
        
        # === MODIFICATION : Utiliser ensure_algo_has_trainset ===
        algo_with_trainset = ensure_algo_has_trainset(algo, train_df)
        prediction = algo_with_trainset.predict(test_user, test_movie)
        # === FIN MODIFICATION ===
        
        print(f"✅ Test prédiction réussi: user={test_user}, movie={test_movie}, rating={prediction.est}")
        return True
        
    except Exception as e:
        print(f"❌ Test prédiction échoué: {e}")
        import traceback
        traceback.print_exc()
        return False