import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create user-movie matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values
user_movie_matrix = user_movie_matrix.fillna(0)

# Calculate similarity between movies
similarity = cosine_similarity(user_movie_matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

# Recommendation function
def recommend_movies(movie_name, num_recommendations=5):
    
    if movie_name not in similarity_df.columns:
        return "Movie not found in database"
    
    similar_scores = similarity_df[movie_name].sort_values(ascending=False)
    
    # Remove the movie itself
    similar_scores = similar_scores.drop(movie_name)
    
    return similar_scores.head(num_recommendations)

# Example
movie = "Titanic"
recommendations = recommend_movies(movie)

print(f"\nMovies similar to '{movie}':\n")
print(recommendations)