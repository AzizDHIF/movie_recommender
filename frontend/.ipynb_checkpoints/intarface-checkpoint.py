import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommender System")
st.markdown("Get personalized movie recommendations based on your user ID")

# Sidebar
with st.sidebar:
    st.header("Settings")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    
    st.header("Options")
    show_details = st.checkbox("Show detailed ratings", value=True)
    num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

# Main interface
tab1, tab2 = st.tabs(["Recommendations", "Custom Prediction"])

with tab1:
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"user_id": int(user_id)}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data["top_recommendations"][:num_recommendations]
                    
                    st.success(f"Found {len(recommendations)} recommendations for User {user_id}")
                    
                    # Display as table
                    df = pd.DataFrame(recommendations)
                    df.index = df.index + 1
                    st.dataframe(df[["title", "predicted_rating"]], 
                                column_config={
                                    "title": "Movie Title",
                                    "predicted_rating": st.column_config.NumberColumn(
                                        "Predicted Rating",
                                        format="%.2f ⭐"
                                    )
                                })
                    
                    # Display as cards
                    if show_details:
                        st.divider()
                        st.subheader("Detailed View")
                        cols = st.columns(3)
                        for idx, movie in enumerate(recommendations):
                            with cols[idx % 3]:
                                st.markdown(f"""
                                **{movie['title']}**  
                                ⭐ {movie['predicted_rating']}/5  
                                ID: {int(movie['movie_id'])}
                                """)
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {e}")

with tab2:
    st.subheader("Predict ratings for specific movies")
    movie_ids_input = st.text_input("Enter Movie IDs (comma-separated)", "1, 2, 3, 4, 5")
    
    if st.button("Predict Ratings"):
        movie_ids = [int(id.strip()) for id in movie_ids_input.split(",")]
        
        with st.spinner("Predicting ratings..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={
                        "user_id": int(user_id),
                        "movie_ids": movie_ids
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "predictions" in data:
                        predictions = data["predictions"]
                        
                        st.success(f"Predictions for User {user_id}")
                        
                        # Create chart
                        chart_data = pd.DataFrame(predictions)
                        st.bar_chart(chart_data.set_index("title")["predicted_rating"])
                        
                        # Display table
                        st.dataframe(chart_data)
                    else:
                        st.warning("No predictions returned")
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {e}")

# Footer
st.divider()
st.markdown("""
**API Endpoints:**
- `POST /predict` - Get movie recommendations
- `GET /health` - API health check
- `GET /docs` - Interactive documentation
""")
