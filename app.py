
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import os
from pathlib import Path
from PIL import Image
from io import BytesIO


# -------------------
# Load saved models
# -------------------
with open(r"D:\Ai course MEC\MEC_final_project_2\Recommendation system\model.pkl", "rb") as f:
    cf_model = pickle.load(f)

with open(r"D:\Ai course MEC\MEC_final_project_2\Recommendation system\model_2.pkl", "rb") as f:
    hybrid_model = pickle.load(f)

with open(r"D:\Ai course MEC\MEC_final_project_2\Recommendation system\movie_id_to_title.pkl", "rb") as f:
    movie_mapping = pickle.load(f)  # movie_id -> title

with open(r"D:\Ai course MEC\MEC_final_project_2\Recommendation system\user_id_map.pkl", "rb") as f:
    user_mapping = pickle.load(f)  # user_id -> index

with open(r"D:\Ai course MEC\MEC_final_project_2\Recommendation system\interactions_cf_shape.pkl", "rb") as f:
    interactions_shape = pickle.load(f)  # (num_users, num_items)

# -------------------
# Helper function to get recommendations
# -------------------
def recommend_movies(model, user_id, num_recommendations=5):
    user_index = user_mapping.get(user_id)
    if user_index is None:
        return ["User not found!"]

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

# Select user
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

    st.subheader(f"Recommendations for User {user_id}:")
    for title, score in recs:
        st.write(f"⭐ {title} (Score: {score:.2f})")
