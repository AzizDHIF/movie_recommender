import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.markdown("Simulation de recommandations évolutives pour un nouvel utilisateur")

# -------------------------
# Entrées utilisateur
# -------------------------
user_id = st.number_input("User ID", min_value=1, step=1)

movie_ids_input = st.text_input(
    "Films notés (IDs séparés par des virgules, optionnel)",
    placeholder="ex: 1, 50, 296"
)

# -------------------------
# Bouton prédiction
# -------------------------
if st.button("🔍 Obtenir des recommandations"):

    payload = {"user_id": int(user_id)}

    if movie_ids_input.strip():
        movie_ids = [
            int(m.strip())
            for m in movie_ids_input.split(",")
            if m.strip().isdigit()
        ]
        payload["movie_ids"] = movie_ids

    with st.spinner("Calcul des recommandations..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()

        if "predictions" in data:
            st.subheader("🎯 Prédictions pour les films fournis")
            for rec in data["predictions"]:
                st.write(
                    f"**{rec['title']}** — ⭐ {rec['predicted_rating']}"
                )

        if "top_recommendations" in data:
            st.subheader("🔥 Top recommandations")
            for rec in data["top_recommendations"]:
                st.write(
                    f"**{rec['title']}** — ⭐ {rec['predicted_rating']}"
                )
    else:
        st.error(f"Erreur API : {response.text}")
