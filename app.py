import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# -------------------
# Paths
# -------------------
BASE_DIR = Path(__file__).parent

# -------------------
# Load saved models
# -------------------
def load_pickle(file_name):
    file_path = BASE_DIR / file_name
    if not file_path.exists():
        st.error(f"Missing file: {file_name}")
        st.stop()
    with open(file_path, "rb") as f:
        return pickle.load(f)

cf_model = load_pickle("model.pkl")
hybrid_model = load_pickle("model_2.pkl")
movie_mapping = load_pickle("movie_id_to_title.pkl")
user_mapping = load_pickle("user_id_map.pkl")
interactions_shape = load_pickle("interactions_cf_shape.pkl")  # tuple: (num_users, num_items)

# -------------------
# Helper function to get recommendations
# -------------------
def recommend_movies(model, user_id, num_recommendations=5):
    user_index = user_mapping.get(user_id)
    if user_index is None:
        return None

    # Predict for all items
    scores = model.predict(user_index, np.arange(interactions_shape[1]))

    # Get top N items
    top_items = np.argsort(-scores)[:num_recommendations]

    recommendations = []
    for item_index in top_items:
        title = movie_mapping.get(item_index, f"Movie {item_index}")
        recommendations.append((title, scores[item_index]))

    return recommendations

# -------------------
# Streamlit UI
# -------------------
st.title("🎬 Movie Recommendation System")
st.write("Powered by LightFM")

user_id = st.number_input("Enter User ID", min_value=1, step=1)

model_choice = st.radio(
    "Choose a model:",
    ("Collaborative Filtering", "Hybrid Filtering")
)

if st.button("Get Recommendations"):
    if model_choice == "Collaborative Filtering":
        recs = recommend_movies(cf_model, user_id)
    else:
        recs = recommend_movies(hybrid_model, user_id)

    if recs is None:
        st.warning(f"User ID {user_id} not found in the dataset.")
    else:
        st.subheader(f"Recommendations for User {user_id}:")
        for title, score in recs:
            st.write(f"⭐ {title} (Score: {score:.2f})")
