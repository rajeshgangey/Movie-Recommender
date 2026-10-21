import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommendations using genres, descriptions, TF-IDF and cosine similarity.")
movies = pd.read_csv("data/movies.csv")
features = (movies["genres"].fillna("") + " " + movies["description"].fillna(""))
matrix = TfidfVectorizer(stop_words="english").fit_transform(features)
similarity = cosine_similarity(matrix)
choice = st.selectbox("Choose a movie you like", movies["title"].tolist())
n = st.slider("Number of recommendations", 1, 5, 3)
if st.button("Recommend"):
    idx = movies.index[movies["title"] == choice][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted((x for x in scores if x[0] != idx), key=lambda x: x[1], reverse=True)[:n]
    result = movies.iloc[[i for i, _ in scores]][["title", "genres", "description"]].copy()
    result.insert(1, "similarity_score", [round(float(score), 3) for _, score in scores])
    st.dataframe(result, use_container_width=True)
with st.expander("Movie catalogue"):
    st.dataframe(movies, use_container_width=True)
