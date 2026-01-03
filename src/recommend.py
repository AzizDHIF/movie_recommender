# src/recommend.py
"""
Module de recommandation avec gestion adaptative du cold start.

Stratégie en 3 niveaux :
- POPULAR : 0 ratings → Films populaires
- HYBRID : 1-9 ratings → Popularité + Genres préférés + SVD
- PERSONALIZED : 10+ ratings → SVD pur (collaborative filtering)
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS POUR ENCODERS
# ============================================================================

def safe_transform(encoder, id_value, default_idx=None):
    """
    Encode un ID de manière sécurisée (compatible dict et LabelEncoder).
    
    Args:
        encoder: LabelEncoder sklearn ou dict
        id_value: ID à encoder
        default_idx: Index par défaut si l'ID n'existe pas
        
    Returns:
        Index encodé
    """
    try:
        if hasattr(encoder, 'transform'):
            # LabelEncoder sklearn
            return encoder.transform([id_value])[0]
        else:
            # Dict Python
            return encoder[id_value]
    except (ValueError, KeyError):
        if default_idx is not None:
            return default_idx
        else:
            raise ValueError(f"ID {id_value} not in encoder")


def get_all_classes(encoder):
    """
    Récupère toutes les classes d'un encoder.
    
    Args:
        encoder: LabelEncoder ou dict
        
    Returns:
        Liste ou array des classes
    """
    if hasattr(encoder, 'classes_'):
        # LabelEncoder
        return encoder.classes_
    else:
        # Dict
        return list(encoder.keys())


def encoder_len(encoder):
    """
    Retourne la longueur d'un encoder.
    
    Args:
        encoder: LabelEncoder ou dict
        
    Returns:
        Nombre d'éléments
    """
    if hasattr(encoder, 'classes_'):
        return len(encoder.classes_)
    else:
        return len(encoder)


def is_in_encoder(encoder, id_value):
    """
    Vérifie si un ID est dans l'encoder.
    
    Args:
        encoder: LabelEncoder ou dict
        id_value: ID à vérifier
        
    Returns:
        True si présent, False sinon
    """
    if hasattr(encoder, 'classes_'):
        # LabelEncoder
        return id_value in encoder.classes_
    else:
        # Dict
        return id_value in encoder


# ============================================================================
# FONCTION PRINCIPALE DE RECOMMANDATION
# ============================================================================

def recommend_movies(user_id: int, 
                     train_df: pd.DataFrame, 
                     df_movies: pd.DataFrame, 
                     algo, 
                     user_encoder, 
                     movie_encoder, 
                     top_n: int = 5) -> List[Tuple[int, str, float]]:
    """
    Génère les top N recommandations pour un utilisateur (stratégie personnalisée).
    
    Compatible avec LabelEncoder sklearn et dict.
    
    Args:
        user_id: ID de l'utilisateur
        train_df: DataFrame des ratings
        df_movies: DataFrame des films
        algo: Modèle Surprise (SVD, etc.)
        user_encoder: Encoder des user IDs
        movie_encoder: Encoder des movie IDs
        top_n: Nombre de recommandations
    
    Returns:
        Liste de tuples (movie_id, title, predicted_rating)
    """
    logger.info(f"Generating personalized recommendations for user {user_id}")
    
    # 1. Encoder l'utilisateur
    try:
        user_idx = safe_transform(user_encoder, user_id)
    except ValueError:
        # Nouvel utilisateur non dans l'encoder
        user_idx = encoder_len(user_encoder)
        logger.warning(f"New user {user_id} not in encoder, assigned idx {user_idx}")
    
    # 2. Films déjà notés par l'utilisateur
    movies_watched = train_df[train_df['userId'] == user_id]['movieId'].tolist()
    
    # 3. Tous les films candidats (non notés)
    all_movie_ids = set(get_all_classes(movie_encoder))
    candidate_movie_ids = all_movie_ids - set(movies_watched)
    
    logger.info(f"User has watched {len(movies_watched)} movies, {len(candidate_movie_ids)} candidates")
    
    # 4. Prédictions pour chaque film candidat
    predictions = []
    
    for movie_id in candidate_movie_ids:
        try:
            movie_idx = safe_transform(movie_encoder, movie_id)
            pred_rating = algo.predict(user_idx, movie_idx).est
            predictions.append((movie_id, pred_rating))
        except (ValueError, Exception) as e:
            # Film pas dans l'encoder ou erreur de prédiction
            logger.debug(f"Skipping movie {movie_id}: {e}")
            continue
    
    # 5. Trier par rating prédit (décroissant)
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]
    
    logger.info(f"Generated {len(top_predictions)} predictions")
    
    # 6. Convertir en titres
    recommendations = []
    for movie_id, rating in top_predictions:
        title = df_movies.loc[
            df_movies['movieId'] == movie_id, 'title'
        ].values
        
        if len(title) > 0:
            recommendations.append((int(movie_id), title[0], float(rating)))
        else:
            logger.warning(f"Movie {movie_id} not found in df_movies")
    
    logger.info(f"✓ Generated {len(recommendations)} personalized recommendations")
    return recommendations


# ============================================================================
# STRATÉGIES ADDITIONNELLES (OPTIONNELLES)
# ============================================================================

def get_popular_movies(df_movies: pd.DataFrame,
                       train_df: pd.DataFrame,
                       top_n: int = 10,
                       min_ratings: int = 50) -> List[Tuple[int, str, float]]:
    """
    Recommande les films les plus populaires (cold start).
    
    Args:
        df_movies: DataFrame des films
        train_df: DataFrame des ratings
        top_n: Nombre de recommandations
        min_ratings: Nombre minimum de ratings
    
    Returns:
        Liste de tuples (movie_id, title, avg_rating)
    """
    logger.info("Generating popular movie recommendations")
    
    # Calculer statistiques par film
    movie_stats = train_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Filtrer films populaires
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    
    # Trier par moyenne décroissante
    popular_movies = popular_movies.sort_values(
        by=['avg_rating', 'num_ratings'], 
        ascending=[False, False]
    ).head(top_n)
    
    # Joindre avec titres
    recommendations = popular_movies.merge(
        df_movies[['movieId', 'title']], 
        on='movieId'
    )
    
    result = [
        (int(row['movieId']), row['title'], float(row['avg_rating']))
        for _, row in recommendations.iterrows()
    ]
    
    logger.info(f"✓ Generated {len(result)} popular recommendations")
    return result


def get_user_preferred_genres(user_id: int,
                               train_df: pd.DataFrame,
                               df_movies: pd.DataFrame,
                               top_k: int = 3) -> List[str]:
    """
    Identifie les genres préférés d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        train_df: DataFrame des ratings
        df_movies: DataFrame des films
        top_k: Nombre de genres à retourner
    
    Returns:
        Liste des genres préférés
    """
    # Récupérer les films bien notés (>= 4.0)
    user_high_ratings = train_df[
        (train_df['userId'] == user_id) & 
        (train_df['rating'] >= 4.0)
    ]
    
    if len(user_high_ratings) == 0:
        return []
    
    # Joindre avec les genres
    user_movies = user_high_ratings.merge(
        df_movies[['movieId', 'genres']], 
        on='movieId',
        how='left'
    )
    
    # Exploser les genres
    all_genres = []
    for genres_str in user_movies['genres'].dropna():
        all_genres.extend(genres_str.split('|'))
    
    if not all_genres:
        return []
    
    # Compter et retourner top K
    genre_counts = pd.Series(all_genres).value_counts()
    return genre_counts.head(top_k).index.tolist()


def recommend_hybrid(user_id: int,
                     train_df: pd.DataFrame,
                     df_movies: pd.DataFrame,
                     algo,
                     user_encoder,
                     movie_encoder,
                     top_n: int = 10,
                     weight_popular: float = 0.3,
                     weight_genres: float = 0.2,
                     weight_collaborative: float = 0.5) -> List[Tuple[int, str, float]]:
    """
    Recommandations hybrides (cold start partiel).
    
    Combine popularité, genres préférés et collaborative filtering.
    
    Args:
        user_id: ID de l'utilisateur
        train_df: DataFrame des ratings
        df_movies: DataFrame des films
        algo: Modèle Surprise
        user_encoder: Encoder users
        movie_encoder: Encoder movies
        top_n: Nombre de recommandations
        weight_popular: Poids popularité
        weight_genres: Poids genres
        weight_collaborative: Poids SVD
    
    Returns:
        Liste de tuples (movie_id, title, hybrid_score)
    """
    logger.info(f"Generating hybrid recommendations for user {user_id}")
    
    # 1. Films déjà notés
    rated_movies = set(train_df[train_df['userId'] == user_id]['movieId'])
    
    # 2. Genres préférés
    preferred_genres = get_user_preferred_genres(user_id, train_df, df_movies)
    logger.info(f"User preferred genres: {preferred_genres}")
    
    # 3. Filtrer candidats par genres (si disponibles)
    if preferred_genres:
        genre_mask = df_movies['genres'].apply(
            lambda x: any(g in str(x) for g in preferred_genres) if pd.notna(x) else False
        )
        candidate_movies = df_movies[genre_mask]
    else:
        candidate_movies = df_movies
    
    # 4. Statistiques de popularité
    movie_stats = train_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    movie_stats['popularity_score'] = (
        movie_stats['num_ratings'] / movie_stats['num_ratings'].max()
    )
    
    # 5. Encoder utilisateur
    try:
        user_idx = safe_transform(user_encoder, user_id)
    except ValueError:
        user_idx = encoder_len(user_encoder)
        logger.warning(f"New user {user_id}, assigned idx {user_idx}")
    
    # 6. Calculer scores hybrides
    hybrid_scores = []
    
    for _, movie_row in candidate_movies.iterrows():
        movie_id = movie_row['movieId']
        
        if movie_id in rated_movies:
            continue
        
        try:
            movie_idx = safe_transform(movie_encoder, movie_id)
        except ValueError:
            continue
        
        # Score SVD
        try:
            svd_prediction = algo.predict(user_idx, movie_idx).est
        except Exception as e:
            logger.debug(f"SVD prediction failed for movie {movie_id}: {e}")
            svd_prediction = 3.0
        
        # Score popularité
        pop_info = movie_stats[movie_stats['movieId'] == movie_id]
        if len(pop_info) > 0:
            popularity_rating = pop_info['avg_rating'].values[0]
        else:
            popularity_rating = 3.0
        
        # Score genres
        movie_genres = str(movie_row['genres']).split('|')
        genre_match = sum(1 for g in preferred_genres if g in movie_genres)
        genre_score = (genre_match / max(len(preferred_genres), 1)) * 5.0
        
        # Score hybride
        hybrid_score = (
            weight_collaborative * svd_prediction +
            weight_popular * popularity_rating +
            weight_genres * genre_score
        )
        
        hybrid_scores.append((movie_id, movie_row['title'], hybrid_score))
    
    # 7. Trier et retourner top N
    hybrid_scores.sort(key=lambda x: x[2], reverse=True)
    result = hybrid_scores[:top_n]
    
    logger.info(f"✓ Generated {len(result)} hybrid recommendations")
    return result


# ============================================================================
# FONCTION ROUTER (OPTIONNELLE)
# ============================================================================

def get_recommendations(user_id: int,
                        train_df: pd.DataFrame,
                        df_movies: pd.DataFrame,
                        algo,
                        user_encoder,
                        movie_encoder,
                        top_n: int = 10) -> List[Tuple[int, str, float]]:
    """
    Router intelligent qui choisit la stratégie selon le profil utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        train_df: DataFrame des ratings
        df_movies: DataFrame des films
        algo: Modèle Surprise
        user_encoder: Encoder users
        movie_encoder: Encoder movies
        top_n: Nombre de recommandations
    
    Returns:
        Liste de tuples (movie_id, title, score)
    """
    # Compter les ratings de l'utilisateur
    num_ratings = len(train_df[train_df['userId'] == user_id])
    
    logger.info(f"User {user_id} has {num_ratings} ratings")
    
    # Choisir la stratégie
    if num_ratings == 0:
        # Cold start total : films populaires
        logger.info("Strategy: POPULAR (cold start)")
        return get_popular_movies(df_movies, train_df, top_n)
    
    elif num_ratings < 10:
        # Cold start partiel : hybride
        logger.info("Strategy: HYBRID (cold start partial)")
        return recommend_hybrid(
            user_id, train_df, df_movies,
            algo, user_encoder, movie_encoder, top_n
        )
    
    else:
        # Utilisateur établi : personnalisé
        logger.info("Strategy: PERSONALIZED")
        return recommend_movies(
            user_id, train_df, df_movies,
            algo, user_encoder, movie_encoder, top_n
        )


# ============================================================================
# CLI POUR TESTS
# ============================================================================

if __name__ == "__main__":
    print("Module recommend.py - Tests unitaires")
    
    # Test des helper functions
    from sklearn.preprocessing import LabelEncoder
    
    print("\n1. Test LabelEncoder...")
    le = LabelEncoder()
    le.fit([10, 20, 30])
    
    assert safe_transform(le, 10) == 0
    assert encoder_len(le) == 3
    assert is_in_encoder(le, 10) == True
    assert is_in_encoder(le, 99) == False
    print("✓ LabelEncoder OK")
    
    print("\n2. Test Dict...")
    d = {10: 0, 20: 1, 30: 2}
    
    assert safe_transform(d, 10) == 0
    assert encoder_len(d) == 3
    assert is_in_encoder(d, 10) == True
    assert is_in_encoder(d, 99) == False
    print("✓ Dict OK")
    
    print("\n✓ All tests passed!")