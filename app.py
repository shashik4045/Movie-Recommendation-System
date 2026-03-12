import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("🎬 Movie Recommendation System")

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create user-movie matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values
user_movie_matrix = user_movie_matrix.fillna(0)

# Calculate similarity
similarity = cosine_similarity(user_movie_matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

# Recommendation function
def recommend_movies(movie_name):
    
    if movie_name not in similarity_df.columns:
        return []
    
    similar_scores = similarity_df[movie_name].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_name)

    return similar_scores.head(5)

# Movie selection
movie_list = user_movie_matrix.columns.tolist()

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

if st.button("Recommend Movies"):
    
    recommendations = recommend_movies(selected_movie)

    st.subheader("Recommended Movies:")

    for movie, score in recommendations.items():
        st.write(f"🎥 {movie} (Similarity: {round(score,2)})")