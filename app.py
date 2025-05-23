import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
links = pd.read_csv("links.csv")

# Preprocess
movies['genres'] = movies['genres'].fillna("")

# TF-IDF on genres
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# Compute average ratings
avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']

# Merge into movies DataFrame
movies = movies.merge(avg_ratings, on='movieId', how='left')
movies = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

# IMDb URL
def get_imdb_link(imdb_id):
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/" if pd.notna(imdb_id) else "N/A"

# Recommendation function 
def get_recommendations(title, top_n=5):
    try:
        idx = movies[movies['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return None

    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
    rec_indices = indices.flatten()[1:]

    recs = movies.iloc[rec_indices].copy()
    recs = recs.sort_values(by='avg_rating', ascending=False)

    recommendations = []
    for _, row in recs.iterrows():
        recommendations.append({
            "Title": row['title'],
            "Genres": row['genres'],
            "Avg Rating": round(row['avg_rating'], 2) if not pd.isna(row['avg_rating']) else "No rating",
            "IMDb Link": get_imdb_link(row['imdbId'])
        })

    return recommendations

# Sidebar: Top Picks by Genre (manually curated)
top_picks = {
    "Action": [
        ("John Wick: Chapter 4", "https://www.imdb.com/title/tt10366206/"),
        ("Extraction 2", "https://www.imdb.com/title/tt12263384/"),
        ("Top Gun: Maverick", "https://www.imdb.com/title/tt1745960/"),
        ("The Batman", "https://www.imdb.com/title/tt1877830/"),
        ("No Time to Die", "https://www.imdb.com/title/tt2382320/")
    ],
    "Comedy": [
        ("The Lost City", "https://www.imdb.com/title/tt13320622/"),
        ("Free Guy", "https://www.imdb.com/title/tt6264654/"),
        ("Barb and Star Go to Vista Del Mar", "https://www.imdb.com/title/tt3797512/"),
        ("Palm Springs", "https://www.imdb.com/title/tt9484998/"),
        ("Glass Onion: A Knives Out Mystery", "https://www.imdb.com/title/tt11564570/")
    ],
    "Drama": [
        ("Oppenheimer", "https://www.imdb.com/title/tt15398776/"),
        ("The Whale", "https://www.imdb.com/title/tt13833688/"),
        ("The Fabelmans", "https://www.imdb.com/title/tt14208870/"),
        ("Women Talking", "https://www.imdb.com/title/tt13669038/"),
        ("The Banshees of Inisherin", "https://www.imdb.com/title/tt11813216/")
    ],
    "Sci-Fi": [
        ("Dune: Part Two", "https://www.imdb.com/title/tt15239678/"),
        ("The Creator", "https://www.imdb.com/title/tt11858890/"),
        ("Tenet", "https://www.imdb.com/title/tt6723592/"),
        ("Everything Everywhere All at Once", "https://www.imdb.com/title/tt6710474/"),
        ("Spider-Man: Across the Spider-Verse", "https://www.imdb.com/title/tt9362722/")
    ]
}

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# Sidebar for Best Picks
st.sidebar.title(" Best Picks by Genre")
for genre, picks in top_picks.items():
    st.sidebar.markdown(f"### 🎭 {genre}")
    for title, url in picks:
        st.sidebar.markdown(f"- [{title}]({url})")

# Main UI
st.title("🎬 Movie Recommender")
st.markdown("Get **5 similar movies** based on your selected movie.")

movie_titles = sorted(movies['title'].unique())
movie_input = st.selectbox("🎥 Select a movie", movie_titles)

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        results = get_recommendations(movie_input)
        if results is None:
            st.error(f"Movie '{movie_input}' not found.")
        else:
            st.success(f"Top {len(results)} recommendations for '{movie_input}':")
            for i, rec in enumerate(results, 1):
                st.markdown(f"**{i}. {rec['Title']}**")
                st.markdown(f"- Genres: {rec['Genres']}")
                st.markdown(f"- Avg Rating:  {rec['Avg Rating']}")
                st.markdown(f"- [IMDb Link]({rec['IMDb Link']})")
                st.markdown("---")
