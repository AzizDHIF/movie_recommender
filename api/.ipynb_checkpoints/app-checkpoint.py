# api/app.py 
"""
Ce correctif gère les LabelEncoder au lieu des dictionnaires.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
from src.load_save_data import load_model_and_encoders_from_gcs, load_data_from_gcs, load_model_and_encoders_local, load_local_all_data
from src.recommend import recommend_movies, get_recommendations
from src.train import train_best_model
from src.check_compatibility_local_cloud import *
from src.recommend import *
import pickle
import os
import time

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
quick_compare_cloud_vs_local()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GESTION DES DONNEES EN TEMPS REEL 
# ============================================================================
RATINGS_FILE = "realtime_ratings.pkl"

def load_realtime_ratings():
    """Charge les ratings temps réel depuis le fichier"""
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

def save_realtime_ratings():
    """Sauvegarde les ratings temps réel"""
    global realtime_ratings
    with open(RATINGS_FILE, 'wb') as f:
        pickle.dump(realtime_ratings, f)

def get_combined_ratings(user_id=None):
    """
    Combine train_df (original) et realtime_ratings pour un utilisateur.
    Priorité aux ratings temps réel.
    """
    global train_df, realtime_ratings
    
    # Filtrer les ratings temps réel pour l'utilisateur
    if user_id:
        realtime_user = realtime_ratings[realtime_ratings['userId'] == user_id]
    else:
        realtime_user = realtime_ratings
    
    # Filtrer train_df pour l'utilisateur (sans les films notés en temps réel)
    if user_id:
        train_user = train_df[train_df['userId'] == user_id]
    else:
        train_user = train_df
    
    # Exclure de train_df les films déjà notés en temps réel
    if not realtime_user.empty:
        rated_movies = set(realtime_user['movieId'])
        train_user = train_user[~train_user['movieId'].isin(rated_movies)]
    
    # Combiner (priorité aux ratings temps réel)
    combined = pd.concat([realtime_user[['userId', 'movieId', 'rating']], 
                         train_user], ignore_index=True)
    
    return combined

def get_user_rating_count(user_id: int) -> int:
    """Compte le nombre total de ratings (train + temps réel)"""
    return len(get_combined_ratings(user_id))

def get_recommendation_strategy(user_id: int) -> str:
    """Détermine la stratégie basée sur les ratings combinés"""
    count = get_user_rating_count(user_id)
    
    if count == 0:
        return "popular"
    elif count < 10:
        return "hybrid"
    else:
        return "personalized"
  

# Charger les ratings temps réel au démarrage
realtime_ratings = load_realtime_ratings()
logger.info(f"Loaded {len(realtime_ratings)} real-time ratings")
# ============================================================================
# CHARGEMENT MODÈLE ET DONNÉES
# ============================================================================
import os

MODE = os.getenv("RUN_MODE", "cloud").lower()  # cloud par défaut
logger.info(f"🚀 Starting API in {MODE.upper()} mode")

try:
    if MODE == "local":
        logger.info("📦 Loading model and data from LOCAL...")
        algo, user_encoder, movie_encoder = load_model_and_encoders_local()
        df, df_movies, train_df = load_local_all_data()
    elif MODE == "cloud":
        logger.info("☁️ Loading model and data from GCS...")
        algo_bytes, user_bytes, movie_bytes = load_model_and_encoders_from_gcs()

        # 🔹 Dé-pickle les objets
        import pickle
        algo = pickle.loads(algo_bytes)
        user_encoder = pickle.loads(user_bytes)
        movie_encoder = pickle.loads(movie_bytes)

        df, df_movies, train_df = load_data_from_gcs()

    else:
        raise ValueError(f"Invalid RUN_MODE: {MODE}")

    logger.info(f"train_df columns = {train_df.columns.tolist()}")

    # 🔍 Vérification compatibilité modèle / données
    check_model_data_compatibility(
        algo, user_encoder, movie_encoder, train_df
    )

    # 🔧 Gestion LabelEncoder ou dictionnaire
    if hasattr(user_encoder, "classes_"):
        num_users = len(user_encoder.classes_)
        num_movies = len(movie_encoder.classes_)
    else:
        num_users = len(user_encoder)
        num_movies = len(movie_encoder)

    logger.info(
        f"✓ Model loaded successfully | "
        f"Users: {num_users} | "
        f"Movies: {num_movies} | "
        f"Ratings: {len(train_df)}"
    )

except Exception as e:
    logger.exception("Failed to load model or data")
    raise


# ============================================================================
# INITIALISER FASTAPI
# ============================================================================

app = FastAPI(title="Movie Recommender API")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_transform(encoder, id_value, default_idx=None):
    """
    Encode un ID de manière sécurisée.
    
    Args:
        encoder: LabelEncoder ou dict
        id_value: ID à encoder
        default_idx: Index par défaut si l'ID n'existe pas
        
    Returns:
        Index encodé
    """
    try:
        if hasattr(encoder, 'transform'):
            # LabelEncoder
            return encoder.transform([id_value])[0]
        else:
            # Dict
            return encoder[id_value]
    except (ValueError, KeyError):
        if default_idx is not None:
            return default_idx
        else:
            raise ValueError(f"ID {id_value} not in encoder")


# ============================================================================
# MODELS PYDANTIC
# ============================================================================

class Request(BaseModel):
    user_id: int
    movie_ids: list[int] = None


class RatingInput(BaseModel):
    user_id: int
    movie_id: int
    rating: float


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Movie Recommender API",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check détaillé"""
    if hasattr(user_encoder, 'classes_'):
        num_users = len(user_encoder.classes_)
        num_movies = len(movie_encoder.classes_)
    else:
        num_users = len(user_encoder)
        num_movies = len(movie_encoder)
    
    return {
        "status": "healthy",
        "model_loaded": algo is not None,
        "data_loaded": train_df is not None,
        "stats": {
            "num_users": int(num_users),
            "num_movies": int(num_movies),
            "num_ratings": int(len(train_df))
        }
    }


# Dans api/app.py
@app.get("/movies/all")
def get_all_movies(limit: int = 1000):
    """Retourne tous les films avec leurs statistiques"""
    try:
        # Calculer les statistiques de rating
        all_ratings = get_combined_ratings()
        movie_stats = all_ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        # Joindre avec les informations des films
        all_movies = df_movies.merge(
            movie_stats,
            on='movieId',
            how='left'
        )
        
        # Remplir les valeurs manquantes
        all_movies['avg_rating'] = all_movies['avg_rating'].fillna(0.0)
        all_movies['num_ratings'] = all_movies['num_ratings'].fillna(0)
        
        # Limiter le nombre de résultats
        all_movies = all_movies.head(limit)
        
        # Convertir en format JSON
        movies_list = []
        for _, row in all_movies.iterrows():
            movies_list.append({
                "movie_id": int(row['movieId']),
                "title": row['title'],
                "genres": row['genres'],
                "avg_rating": float(row['avg_rating']),
                "num_ratings": int(row['num_ratings'])
            })
        
        return {
            "total_movies": len(df_movies),
            "returned_movies": len(movies_list),
            "movies": movies_list
        }
        
    except Exception as e:
        logger.error(f"Error in /movies/all: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict")
def predict(req: Request):
    """
    Génère des recommandations avec stratégie adaptative.
    """
    try:
        # Déterminer la stratégie AVEC LES DONNÉES COMBINÉES
        user_combined_ratings = get_combined_ratings(req.user_id)  
        num_ratings = len(user_combined_ratings) 
        strategy = get_recommendation_strategy(req.user_id)
        
        logger.info(
            f"Request: user={req.user_id}, ratings={num_ratings}, strategy={strategy}"
        )
        if strategy == "popular":
            logger.info(f"Using POPULAR strategy for user {req.user_id}")
            recommendations_tuples = get_popular_movies(
            df_movies,
            train_df,   
            top_n=20
           )
            
        elif strategy == "hybrid":
            logger.info(f"Using HYBRID strategy for user {req.user_id}")
            recommendations_tuples = recommend_hybrid(
                req.user_id, user_combined_ratings, df_movies,
                algo, user_encoder, movie_encoder, top_n=20
            )
            
        else:  # "personalized"
            logger.info(f"Using PERSONALIZED SVD for user {req.user_id}")
            recommendations_tuples = recommend_movies(
                req.user_id, user_combined_ratings, df_movies,
                algo, user_encoder, movie_encoder, top_n=20
            )
        
        # Formater les résultats
        recommendations = []
        for movie_id, title, rating in recommendations_tuples:
            genres = df_movies[df_movies['movieId'] == movie_id]['genres'].values
            genres_str = genres[0] if len(genres) > 0 else None
            recommendations.append({
                "movie_id": int(movie_id),
                "title": title,
                "predicted_rating": round(float(rating), 2),
                "genres": genres_str
            })
        
        return {
            "user_id": req.user_id,
            "num_ratings": num_ratings,  # ← Nombre réel de ratings
            "recommendation_strategy": strategy,
            "recommendations": recommendations
        }
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}", exc_info=True)
        
        if strategy == "popular":
            # Cold start total : films populaires
            logger.info(f"Using POPULAR strategy for user {req.user_id}")
            recommendations_tuples = get_popular_movies(df_movies, train_df, top_n=20)
            
        elif strategy == "hybrid":
            # Cold start partiel : hybride
            logger.info(f"Using HYBRID strategy for user {req.user_id}")
            recommendations_tuples = recommend_hybrid(
                req.user_id, train_df, df_movies,
                algo, user_encoder, movie_encoder, top_n=20
            )
            
        else:  # "personalized"
            # Utilisateur établi : SVD personnalisé
            logger.info(f"Using PERSONALIZED SVD for user {req.user_id}")
            recommendations_tuples = recommend_movies(
                req.user_id, train_df, df_movies,
                algo, user_encoder, movie_encoder, top_n=20
            )
        
        # Formater les résultats
        recommendations = []
        for movie_id, title, rating in recommendations_tuples:
            genres = df_movies[df_movies['movieId'] == movie_id]['genres'].values
            genres_str = genres[0] if len(genres) > 0 else None
            recommendations.append({
                "movie_id": int(movie_id),
                "title": title,
                "predicted_rating": round(float(rating), 2),
                "genres":genres_str
            })
        
        return {
            "user_id": req.user_id,
            "num_ratings": num_ratings,
            "recommendation_strategy": strategy,  
            "recommendations": recommendations
        }
        # ============================================================
        # FIN DU REMPLACEMENT
        # ============================================================
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}", exc_info=True)
        
        # Fallback SEULEMENT en cas d'erreur
        try:
            logger.warning(f"Using fallback for user {req.user_id} due to error: {e}")
            
            # Obtenir les films déjà notés
            rated_movies = set(train_df[train_df['userId'] == req.user_id]['movieId'])
            
            # Obtenir les films populaires non notés
            movie_stats = train_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
            
            # Filtrer et trier
            popular_movies = movie_stats[
                (~movie_stats['movieId'].isin(rated_movies)) &
                (movie_stats['num_ratings'] >= 10)
            ]
            popular_movies = popular_movies.sort_values(
                by=['avg_rating', 'num_ratings'], 
                ascending=[False, False]
            ).head(5)
            
            # Formater les résultats du fallback
            fallback_recommendations = []
            for _, row in popular_movies.iterrows():
                movie_id = row['movieId']
                title = df_movies[df_movies['movieId'] == movie_id]['title'].iloc[0]
                
                fallback_recommendations.append({
                    "movie_id": int(movie_id),
                    "title": title,
                    "predicted_rating": round(float(row['avg_rating']), 2)
                })
            
            return {
                "user_id": req.user_id,
                "num_ratings": get_user_rating_count(req.user_id),
                "recommendation_strategy": "popular_fallback_error",
                "error": str(e),
                "recommendations": fallback_recommendations
            }
            
        except Exception as fallback_error:
            # Dernier recours
            logger.error(f"Fallback also failed: {fallback_error}")
            return {
                "user_id": req.user_id,
                "error": f"Main error: {str(e)}, Fallback error: {str(fallback_error)}",
                "recommendations": []
            }
        

@app.post("/rate", status_code=201)
def add_rating(rating: RatingInput):
    """
    Ajoute un rating en temps réel (sans modifier train_df original).
    Stockage persistant dans realtime_ratings.pkl
    """
    global realtime_ratings
    
    try:
        # 1. Vérifier si le film existe
        if rating.movie_id not in df_movies['movieId'].values:
            raise HTTPException(
                status_code=404,
                detail=f"Movie ID {rating.movie_id} not found"
            )
        
        # 2. Vérifier la note
        if not (0.5 <= rating.rating <= 5.0):
            raise HTTPException(
                status_code=400,
                detail="Rating must be between 0.5 and 5.0"
            )
        
        # 3. Créer un nouveau rating avec timestamp
        new_rating = {
            'userId': rating.user_id,
            'movieId': rating.movie_id,
            'rating': rating.rating,
            'timestamp': time.time()
        }
        
        # 4. Ajouter au stockage temps réel
        new_row = pd.DataFrame([new_rating])
        
        if realtime_ratings.empty:
            realtime_ratings = new_row
        else:
            # Supprimer d'abord l'ancien rating si existe
            mask = ~((realtime_ratings['userId'] == rating.user_id) & 
                    (realtime_ratings['movieId'] == rating.movie_id))
            realtime_ratings = realtime_ratings[mask]
            
            # Ajouter le nouveau
            realtime_ratings = pd.concat([realtime_ratings, new_row], ignore_index=True)
        
        # 5. Sauvegarder de manière persistante
        save_realtime_ratings()
        
        # 6. Déterminer l'action (ajouté ou mis à jour)
        original_exists = ((train_df['userId'] == rating.user_id) & 
                          (train_df['movieId'] == rating.movie_id)).any()
        
        action = "updated" if original_exists else "added"
        
        # 7. Infos du film
        movie_title = df_movies[
            df_movies['movieId'] == rating.movie_id
        ]['title'].values[0]
        
        # 8. Calculer les statistiques avec données combinées
        combined_ratings = get_combined_ratings(rating.user_id)
        user_ratings = len(combined_ratings)
        strategy = get_recommendation_strategy(rating.user_id)
        
        logger.info(
            f"Rating {action} (real-time): user={rating.user_id}, "
            f"movie={rating.movie_id}, rating={rating.rating}, "
            f"total={user_ratings}, strategy={strategy}"
        )
        
        return {
            "status": "success",
            "action": action,
            "user_id": rating.user_id,
            "movie_id": rating.movie_id,
            "movie_title": movie_title,
            "rating": rating.rating,
            "total_user_ratings": user_ratings,
            "current_strategy": strategy,
            "storage": "real-time (persistent)",
            "message": "Rating saved. Will be used for real-time recommendations."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /rate: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/user/{user_id}/ratings")
def get_user_ratings(user_id: int):
    """Récupère l'historique de ratings d'un utilisateur (train + temps réel)"""
    try:
        combined_ratings = get_combined_ratings(user_id)
        
        if len(combined_ratings) == 0:
            return {
                "user_id": user_id,
                "num_ratings": 0,
                "ratings": [],
                "strategy": "popular"
            }
        
        # Joindre avec titres et genres
        ratings_with_info = combined_ratings.merge(
            df_movies[['movieId', 'title', 'genres']], 
            on='movieId',
            how='left'
        )
        
        # Convertir en format JSON
        ratings_list = []
        for _, row in ratings_with_info.iterrows():
            rating_dict = {
                "movieId": int(row['movieId']),
                "title": row['title'],
                "rating": float(row['rating']),
                "genres": row['genres'] if pd.notna(row['genres']) else "Unknown"
            }
            ratings_list.append(rating_dict)
        
        return {
            "user_id": user_id,
            "num_ratings": len(combined_ratings),
            "average_rating": round(float(combined_ratings['rating'].mean()), 2),
            "current_strategy": get_recommendation_strategy(user_id),
            "ratings": ratings_list
        }
    
    except Exception as e:
        logger.error(f"Error in /user/{user_id}/ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/retrain")
def retrain():
    """Réentraîne le modèle"""
    global algo, user_encoder, movie_encoder
    
    try:
        logger.info("Starting model retraining...")
        algo, user_encoder, movie_encoder = train_best_model(train_df, save_mode="cloud")
        logger.info("✓ Model retrained")
        
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "num_ratings": len(train_df)
        }
    
    except Exception as e:
        logger.error(f"Error in /retrain: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_statistics():
    """Statistiques globales (incluant realtime_ratings)"""
    try:
        all_ratings = get_combined_ratings()
        
        if hasattr(user_encoder, 'classes_'):
            num_users = len(user_encoder.classes_)
            num_movies = len(movie_encoder.classes_)
        else:
            num_users = len(user_encoder)
            num_movies = len(movie_encoder)
        
        return {
            "total_users": int(num_users),
            "total_movies": int(num_movies),
            "total_ratings": int(len(all_ratings)),
            "real_time_ratings": int(len(realtime_ratings)),
            "average_rating": float(all_ratings['rating'].mean()),
            "rating_distribution": all_ratings['rating'].value_counts().to_dict()
        }
    
    except Exception as e:
        logger.error(f"Error in /stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))