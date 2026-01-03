# api/app.py - CORRECTIF RAPIDE pour LabelEncoder
"""
Ce correctif gère les LabelEncoder au lieu des dictionnaires.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging

from src.load_save_data import load_model_and_encoders_from_gcs, load_data_from_gcs
from src.recommend import recommend_movies
from src.train import train_best_model

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CHARGEMENT MODÈLE ET DONNÉES
# ============================================================================

logger.info("Loading model and data from GCS...")
try:
    algo, user_encoder, movie_encoder = load_model_and_encoders_from_gcs()
    train_df, df_movies, _ = load_data_from_gcs()  # ✅ Récupérer les 3 éléments
    
    # FIX: Gérer LabelEncoder
    if hasattr(user_encoder, 'classes_'):
        num_users = len(user_encoder.classes_)
        num_movies = len(movie_encoder.classes_)
    else:
        num_users = len(user_encoder)
        num_movies = len(movie_encoder)
    
    logger.info(
        f"✓ Model loaded | Users: {num_users} | "
        f"Movies: {num_movies} | Ratings: {len(train_df)}"  # ✅ Maintenant train_df est un DataFrame
    )
except Exception as e:
    logger.error(f"Failed to load model/data: {e}")
    raise

# ============================================================================
# INITIALISER FASTAPI
# ============================================================================

app = FastAPI(title="Movie Recommender API")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_rating_count(user_id: int) -> int:
    """Compte le nombre de ratings d'un utilisateur"""
    return len(train_df[train_df['userId'] == user_id])


def get_recommendation_strategy(user_id: int) -> str:
    """
    Détermine la stratégie de recommandation.
    
    Returns:
        - "popular" : 0 ratings
        - "hybrid" : 1-9 ratings
        - "personalized" : 10+ ratings
    """
    count = get_user_rating_count(user_id)
    
    if count == 0:
        return "popular"
    elif count < 10:
        return "hybrid"
    else:
        return "personalized"


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


@app.post("/predict")
def predict(req: Request):
    """
    Génère des recommandations avec stratégie adaptative.
    """
    try:
        # Déterminer la stratégie
        num_ratings = get_user_rating_count(req.user_id)
        strategy = get_recommendation_strategy(req.user_id)
        
        logger.info(
            f"Request: user={req.user_id}, ratings={num_ratings}, strategy={strategy}"
        )
        
        # Si films fournis, les ajouter
        if req.movie_ids:
            global train_df
            for movie_id in req.movie_ids:
                # Vérifier si pas déjà noté
                existing = train_df[
                    (train_df['userId'] == req.user_id) & 
                    (train_df['movieId'] == movie_id)
                ]
                
                if len(existing) == 0:
                    new_row = pd.DataFrame([{
                        'userId': req.user_id,
                        'movieId': movie_id,
                        'rating': 4.0
                    }])
                    train_df = pd.concat([train_df, new_row], ignore_index=True)
            
            return {
                "status": "success",
                "user_id": req.user_id,
                "movies_added": req.movie_ids,
                "total_ratings": get_user_rating_count(req.user_id)
            }
        
        # Sinon, générer des recommandations
        recommendations = recommend_movies(
            req.user_id, 
            train_df, 
            df_movies, 
            algo, 
            user_encoder, 
            movie_encoder, 
            top_n=5
        )
        
        return {
            "user_id": req.user_id,
            "num_ratings": num_ratings,
            "recommendation_strategy": strategy,
            "recommendations": [
                {
                    "movie_id": int(m),
                    "title": t,
                    "predicted_rating": round(float(r), 2)
                }
                for m, t, r in recommendations
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rate", status_code=201)
def add_rating(rating: RatingInput):
    """
    Ajoute ou met à jour un rating.
    """
    global train_df
    
    try:
        # Vérifier si le film existe
        if rating.movie_id not in df_movies['movieId'].values:
            raise HTTPException(
                status_code=404,
                detail=f"Movie ID {rating.movie_id} not found"
            )
        
        # Vérifier si déjà noté
        existing = train_df[
            (train_df['userId'] == rating.user_id) & 
            (train_df['movieId'] == rating.movie_id)
        ]
        
        if len(existing) > 0:
            # Mettre à jour
            train_df.loc[existing.index, 'rating'] = rating.rating
            action = "updated"
        else:
            # Ajouter
            new_row = pd.DataFrame([{
                'userId': rating.user_id,
                'movieId': rating.movie_id,
                'rating': rating.rating
            }])
            train_df = pd.concat([train_df, new_row], ignore_index=True)
            action = "added"
        
        # Infos du film
        movie_title = df_movies[
            df_movies['movieId'] == rating.movie_id
        ]['title'].values[0]
        
        # Stats utilisateur
        user_ratings = get_user_rating_count(rating.user_id)
        strategy = get_recommendation_strategy(rating.user_id)
        
        logger.info(
            f"Rating {action}: user={rating.user_id}, movie={rating.movie_id}, "
            f"rating={rating.rating}, total={user_ratings}, strategy={strategy}"
        )
        
        return {
            "status": "success",
            "action": action,
            "user_id": rating.user_id,
            "movie_id": rating.movie_id,
            "movie_title": movie_title,
            "rating": rating.rating,
            "total_user_ratings": user_ratings,
            "current_strategy": strategy
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /rate: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/ratings")
def get_user_ratings(user_id: int):
    """Récupère l'historique de ratings d'un utilisateur"""
    try:
        user_ratings = train_df[train_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return {
                "user_id": user_id,
                "num_ratings": 0,
                "ratings": [],
                "strategy": "popular"
            }
        
        # Joindre avec titres
        ratings_with_info = user_ratings.merge(
            df_movies[['movieId', 'title', 'genres']], 
            on='movieId',
            how='left'
        )
        
        return {
            "user_id": user_id,
            "num_ratings": len(user_ratings),
            "average_rating": round(float(user_ratings['rating'].mean()), 2),
            "current_strategy": get_recommendation_strategy(user_id),
            "ratings": ratings_with_info.to_dict('records')
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
    """Statistiques globales"""
    try:
        if hasattr(user_encoder, 'classes_'):
            num_users = len(user_encoder.classes_)
            num_movies = len(movie_encoder.classes_)
        else:
            num_users = len(user_encoder)
            num_movies = len(movie_encoder)
        
        return {
            "total_users": int(num_users),
            "total_movies": int(num_movies),
            "total_ratings": int(len(train_df)),
            "average_rating": float(train_df['rating'].mean()),
            "rating_distribution": train_df['rating'].value_counts().to_dict()
        }
    
    except Exception as e:
        logger.error(f"Error in /stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))