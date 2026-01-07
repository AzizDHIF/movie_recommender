import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import time
import logging

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

def refresh_user_data():
    """Actualise les données utilisateur après un rating"""
    try:
        # 1. Récupérer les nouveaux ratings
        response = requests.get(f"{API_URL}/user/{st.session_state.user_id}/ratings")
        if response.status_code == 200:
            user_data = response.json()
            st.session_state.user_ratings = user_data.get('ratings', [])
            
            # Mettre à jour dans la sidebar aussi
            st.session_state.num_ratings = user_data.get('num_ratings', 0)
            st.session_state.average_rating = user_data.get('average_rating', 0)
            st.session_state.current_strategy = user_data.get('current_strategy', 'popular')
        
        # 2. Effacer les anciennes recommandations (optionnel)
        # st.session_state.recommendations = []
        
        logger.info(f"Refreshed data for user {st.session_state.user_id}")
        
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        
# ============================================================================
# STYLES CSS PERSONNALISÉS
# ============================================================================

st.markdown("""
<style>
    /* Couleurs principales */
    :root {
        --primary-color: #e50914;
        --secondary-color: #564d4d;
        --background-dark: #141414;
        --card-background: #2f2f2f;
    }
    
    /* Titre principal */
.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(229, 9, 20, 0.8),
                 0 0 40px rgba(229, 9, 20, 0.6),
                 2px 2px 4px rgba(0, 0, 0, 0.5);
}
    
    /* Sous-titre */
    .subtitle {
        text-align: center;
        color: #aaa;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Carte de film */
    .movie-card {
        background: linear-gradient(145deg, #2a2a2a, #1f1f1f);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 1px solid #444;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(229, 9, 20, 0.4);
        border-color: #e50914;
    }
    
    /* Badge de stratégie */
    .strategy-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .badge-popular {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-hybrid {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-personalized {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Étoiles de notation */
    .rating-stars {
        color: #ffd700;
        font-size: 1.2rem;
    }
    
    /* Statistiques */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }

    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(229, 9, 20, 0.4);
    }
    
    /* Genres tags */
    .genre-tag {
        display: inline-block;
        background: rgba(229, 9, 20, 0.2);
        color: #ff6b6b;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 0.2rem;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    div[data-testid="stMarkdownContainer"] p {
    font-size: 20px;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION API
# ============================================================================

API_URL = "http://127.0.0.1:8000"

# ============================================================================
# FONCTIONS HELPER
# ============================================================================

def get_strategy_badge(strategy: str) -> str:
    """Génère un badge HTML pour la stratégie"""
    badges = {
        "popular": ('<span class="strategy-badge badge-popular">🔥 POPULAR</span>', 
                   "Films populaires recommandés"),
        "hybrid": ('<span class="strategy-badge badge-hybrid">🎭 HYBRID</span>',
                  "Recommandations basées sur vos préférences et la popularité"),
        "personalized": ('<span class="strategy-badge badge-personalized">✨ PERSONALIZED</span>',
                        "Recommandations hautement personnalisées")
    }
    return badges.get(strategy, badges["popular"])

def render_stars(rating: float) -> str:
    """Génère des étoiles pour une note"""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    
    stars = "⭐" * full_stars
    if half_star:
        stars += "✨"
    stars += "☆" * empty_stars
    
    return f'<span class="rating-stars">{stars} {rating:.1f}/5</span>'

def extract_genres(title: str) -> List[str]:
    """Extrait les genres du titre (simplifié)"""
    # Dans la vraie version, utilisez la colonne genres du DataFrame
    return []

def format_movie_card(movie: Dict, rank: int) -> str:
    """Génère le HTML d'une carte de film"""
    stars = render_stars(movie.get('predicted_rating', 0))
    
    return f"""
    <div class="movie-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <h3 style="margin: 0; color: #fff; font-size: 1.3rem;">
                    #{rank} {movie['title']}
                </h3>
                <p style="color: #aaa; margin: 0.5rem 0; font-size: 0.9rem;">
                    Movie ID: {movie['movie_id']}
                </p>
                {stars}
            </div>
        </div>
    </div>
    """

# ============================================================================
# SESSION STATE
# ============================================================================

if 'user_id' not in st.session_state:
    st.session_state.user_id = 1

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = []

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-title">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your next favorite movie with AI-powered recommendations</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - CONFIGURATION UTILISATEUR
# ============================================================================

with st.sidebar:
    st.markdown("### 👤 User Profile")
    
    user_id_input = st.number_input(
        "User ID",
        min_value=1,
        max_value=1000,
        value=st.session_state.user_id,
        step=1,
        help="Enter your user ID to get personalized recommendations"
    )
    
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.session_state.recommendations = []
        st.session_state.user_ratings = []
    
    st.markdown("---")
    
    # Statistiques utilisateur
    st.markdown("### 📊 Your Statistics")
    
    try:
        response = requests.get(f"{API_URL}/user/{st.session_state.user_id}/ratings")
        if response.status_code == 200:
            user_data = response.json()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <p class="stat-number">{user_data['num_ratings']}</p>
                    <p class="stat-label">Ratings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_rating = user_data.get('average_rating', 0)
                st.markdown(f"""
                <div class="stat-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <p class="stat-number">{avg_rating:.1f}</p>
                    <p class="stat-label">Avg Rating</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Stratégie actuelle
            st.markdown("### 🎯 Current Strategy")
            strategy = user_data.get('current_strategy', 'popular')
            badge, description = get_strategy_badge(strategy)
            st.markdown(badge, unsafe_allow_html=True)
            st.caption(description)
            
            # Stocker les ratings
            st.session_state.user_ratings = user_data.get('ratings', [])
            
    except Exception as e:
        st.error(f"Could not fetch user data: {e}")
    
    st.markdown("---")
    
    # Options
    st.markdown("### ⚙️ Options")
    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # show_genres = st.checkbox("Show genres", value=True)
    # show_movie_id = st.checkbox("Show movie IDs", value=False)

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Get Recommendations",
    "⭐ Rate Movies",
    "📊 My Ratings",
    "📈 Analytics"
])

# ============================================================================
# TAB 1 : RECOMMANDATIONS
# ============================================================================

with tab1:
    st.markdown("### 🎯 Personalized Recommendations")
    
    # Bouton centré et stylé
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <style>
        div.stButton > button {
            background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 1rem 3rem;
            border-radius: 50px;
            border: none;
            box-shadow: 0 8px 20px rgba(229, 9, 20, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(229, 9, 20, 0.6);
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🎬 GET RECOMMENDATIONS", type="primary", use_container_width=True, key="main_reco_btn"):
            with st.spinner("🔮 Analyzing your preferences..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"user_id": int(st.session_state.user_id)}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        # PAS DE LIMITE ICI - utiliser num_recommendations du slider
                        st.session_state.recommendations = data.get("recommendations", [])
                        
                        # Afficher la stratégie
                        strategy = data.get("recommendation_strategy", "popular")
                        num_ratings = data.get("num_ratings", 0)
                        
                        badge, description = get_strategy_badge(strategy)
                        
                        st.success(f"✨ Found {len(st.session_state.recommendations)} recommendations!")
                        st.markdown(f"""
                        **Strategy used:** {badge}  
                        **Based on:** {num_ratings} ratings  
                        {description}
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection error: {e}")
    
    # Afficher les recommandations en TABLEAU SIMPLE
    if st.session_state.recommendations:
        st.markdown("---")
        st.markdown("### 🎥 Your Recommendations")
        
        # Limiter selon le slider
        recs_to_show = st.session_state.recommendations[:num_recommendations]
        
        # Créer le DataFrame avec genres et IDs
        df_display = pd.DataFrame(recs_to_show)
        
        # Ajouter les genres depuis df_movies si disponible
        if 'genres' not in df_display.columns:
            # Essayer de récupérer les genres
            try:
                response_movies = requests.get(f"{API_URL}/stats")
                # On peut pas facilement récupérer les genres ici, donc on les laisse vides
                df_display['genres'] = 'N/A'
            except:
                df_display['genres'] = 'N/A'
        
        # Renommer et réorganiser les colonnes
        df_display = df_display.rename(columns={
            'movie_id': 'ID',
            'title': 'Title',
            'predicted_rating': 'Rating',
            'genres': 'Genres'
        })
        
        # Sélectionner les colonnes à afficher
        columns_to_show = ['ID', 'Title', 'Genres', 'Rating']
        df_display = df_display[[col for col in columns_to_show if col in df_display.columns]]
        
        # Afficher le tableau
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("Movie ID", width="small"),
                "Title": st.column_config.TextColumn("Movie Title", width="large"),
                "Genres": st.column_config.TextColumn("Genres", width="medium"),
                "Rating": st.column_config.NumberColumn(
                    "Predicted Rating",
                    format="⭐ %.2f",
                    width="small"
                )
            }
        )
        
        # Graphique des ratings prédits
        st.markdown("---")
        st.markdown("### 📊 Ratings Visualization")
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"{r['title'][:30]}..." if len(r['title']) > 30 else r['title'] 
                   for r in recs_to_show],
                y=[r['predicted_rating'] for r in recs_to_show],
                marker=dict(
                    color=[r['predicted_rating'] for r in recs_to_show],
                    colorscale='Reds',
                    line=dict(color='rgb(229, 9, 20)', width=2)
                ),
                text=[f"{r['predicted_rating']:.2f}" for r in recs_to_show],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Predicted Ratings Distribution',
            xaxis_title='Movie',
            yaxis_title='Predicted Rating',
            xaxis_tickangle=-45,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(range=[0, 5.5])
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2 : NOTER DES FILMS (Version Simple)
# ============================================================================

with tab2:
    st.markdown("### ⭐ Rate Movies")
    st.markdown("Rate movies to improve your recommendations!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_id_to_rate = st.number_input(
            "Movie ID",
            min_value=1,
            value=1,
            step=1,
            help="Enter the movie ID you want to rate"
        )
    
    with col2:
        rating_value = st.select_slider(
            "Rating",
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=3.0,
            format_func=lambda x: f"{x} {'⭐' * int(x)}"
        )
    
    if st.button("📝 Submit Rating", type="primary", use_container_width=True):
        with st.spinner("Saving your rating..."):
            try:
                response = requests.post(
                    f"{API_URL}/rate",
                    json={
                        "user_id": int(st.session_state.user_id),
                        "movie_id": int(movie_id_to_rate),
                        "rating": float(rating_value)
                    }
                )
                
                if response.status_code == 201:
                    # Actualiser les données locales
                    refresh_user_data()
                    data = response.json()
                    
                    st.success(f"✅ Rating saved!")
                    st.info(f"""
                    **Movie:** {data['movie_title']}  
                    **Your rating:** {data['rating']} ⭐  
                    **Total ratings:** {data['total_user_ratings']}  
                    **New strategy:** {data['current_strategy']}
                    """)
                    
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {e}")
    
    # Section : Films recommandés à noter
    if st.session_state.recommendations:
        st.markdown("---")
        st.markdown("### 🎯 Quick Rate Recommendations")
        st.markdown("Rate these recommended movies directly:")
        
        for movie in st.session_state.recommendations[:5]:
            with st.expander(f"⭐ {movie['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Predicted Rating:** {movie['predicted_rating']:.1f} ⭐")
                    st.markdown(f"**Movie ID:** {movie['movie_id']}")
                
                with col2:
                    quick_rating = st.selectbox(
                        "Rate it",
                        options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                        key=f"rate_{movie['movie_id']}",
                        label_visibility="collapsed"
                    )
                    
                    if st.button("Save", key=f"btn_{movie['movie_id']}"):
                        try:
                            response = requests.post(
                                f"{API_URL}/rate",
                                json={
                                    "user_id": int(st.session_state.user_id),
                                    "movie_id": int(movie['movie_id']),
                                    "rating": float(quick_rating)
                                }
                            )
                            
                            if response.status_code == 201:
                                st.success("✅ Saved!")
                                time.sleep(0.5)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

# ============================================================================
# TAB 3 : MES NOTES
# ============================================================================

with tab3:
    st.markdown("### 📚 Your Movie Ratings")
        # Vérifier si les données sont à jour
    if not st.session_state.user_ratings:
        try:
            response = requests.get(f"{API_URL}/user/{st.session_state.user_id}/ratings")
            if response.status_code == 200:
                user_data = response.json()
                st.session_state.user_ratings = user_data.get('ratings', [])
        except:
            st.warning("Could not load ratings")
            
    if st.session_state.user_ratings:
        # Filtres
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search movies", "")
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=["rating", "title"],
                format_func=lambda x: "Rating" if x == "rating" else "Title"
            )
        
        # Filtrer et trier
        df_ratings = pd.DataFrame(st.session_state.user_ratings)
        
        if search_term:
            df_ratings = df_ratings[
                df_ratings['title'].str.contains(search_term, case=False, na=False)
            ]
        
        df_ratings = df_ratings.sort_values(by=sort_by, ascending=False)
        
        # Afficher
        st.markdown(f"**Total:** {len(df_ratings)} movies rated")
        
        for idx, row in df_ratings.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['title']}**")
                    if 'genres' in row and pd.notna(row['genres']):
                        genres = row['genres'].split('|')[:3]
                        genres_html = ''.join([f'<span class="genre-tag">{g}</span>' for g in genres])
                        st.markdown(genres_html, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(render_stars(row['rating']), unsafe_allow_html=True)
                
                with col3:
                    st.caption(f"ID: {row['movieId']}")
                
                st.markdown("---")
    else:
        st.info("You haven't rated any movies yet. Start rating to get personalized recommendations!")


# ============================================================================
# TAB 4 : ANALYTICS
# ============================================================================

with tab4:
    st.markdown("### 📈 Your Movie Analytics")
    
    if st.session_state.user_ratings:
        df_ratings = pd.DataFrame(st.session_state.user_ratings)
        
        # Statistiques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Ratings", len(df_ratings))
        
        with col2:
            st.metric("Average Rating", f"{df_ratings['rating'].mean():.2f}")
        
        with col3:
            st.metric("Highest Rating", f"{df_ratings['rating'].max():.1f}")
        
        with col4:
            st.metric("Lowest Rating", f"{df_ratings['rating'].min():.1f}")
        
        st.markdown("---")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des notes
            fig_dist = px.histogram(
                df_ratings,
                x='rating',
                nbins=10,
                title='Rating Distribution',
                labels={'rating': 'Rating', 'count': 'Number of Movies'},
                color_discrete_sequence=['#e50914']
            )
            
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Genres préférés
            if 'genres' in df_ratings.columns:
                # Exploser les genres
                all_genres = []
                for genres_str in df_ratings['genres'].dropna():
                    all_genres.extend(genres_str.split('|'))
                
                if all_genres:
                    genre_counts = pd.Series(all_genres).value_counts().head(10)
                    
                    fig_genres = px.bar(
                        x=genre_counts.values,
                        y=genre_counts.index,
                        orientation='h',
                        title='Your Favorite Genres',
                        labels={'x': 'Number of Movies', 'y': 'Genre'},
                        color=genre_counts.values,
                        color_continuous_scale='Reds'
                    )
                    
                    fig_genres.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_genres, use_container_width=True)
    else:
        st.info("No data to analyze yet. Start rating movies!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p><strong>🎬 Movie Recommender System</strong></p>
    <p>Powered by AI • Built with FastAPI & Streamlit</p>
    <p style="font-size: 0.85rem;">
        <a href="http://127.0.0.1:8000/docs" target="_blank" style="color: #e50914;">API Documentation</a> •
        <a href="http://127.0.0.1:8000/health" target="_blank" style="color: #e50914;">API Health</a>
    </p>
</div>
""", unsafe_allow_html=True)