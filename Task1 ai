import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample dataset
data = {
    'title': ['The Matrix', 'John Wick', 'The Notebook', 'Avengers', 'Titanic'],
    'genre': ['Action Sci-Fi', 'Action Thriller', 'Romance Drama', 'Action Adventure', 'Romance Drama']
}

df = pd.DataFrame(data)

# Convert genre text to TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build a mapping of movie title to index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Test recommendation
print("Recommendations for 'The Matrix':")
print(recommend('The Matrix'))
