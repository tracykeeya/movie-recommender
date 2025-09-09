import streamlit as st
import pandas as pd
import joblib

# ---------- Cached loaders (fast & memory-friendly) ----------
@st.cache_resource
def load_model(path: str):
    # If your model is big, this ensures it's loaded only once
    return joblib.load(path)

@st.cache_data
def load_movies(path: str):
    df = pd.read_csv(path)
    # sanity: keep only expected columns if present
    expected = [c for c in ["movieId", "title", "genres"] if c in df.columns]
    return df[expected] if expected else df

# ---------- App Setup ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Enter a user ID to get top-N movie recommendations.")

# ---------- Load assets ----------
try:
    model = load_model("svd_model.pkl")
except Exception as e:
    st.error(f"Could not load model file 'svd_model.pkl'. Error: {e}")
    st.stop()

try:
    movies = load_movies("movies.csv")
except Exception as e:
    st.error(f"Could not load 'movies.csv'. Error: {e}")
    st.stop()

if "movieId" not in movies.columns:
    st.error("Your movies.csv must have a 'movieId' column.")
    st.stop()

# ---------- UI controls ----------
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("User ID", min_value=1, step=1, value=1)
with col2:
    top_n = st.slider("How many recommendations?", min_value=5, max_value=50, value=10, step=5)

go = st.button("Get Recommendations")

# ---------- Recommender ----------
def recommend_movies(user_id: int, top_n: int = 10) -> pd.DataFrame:
    # Predict a score for every movie
    movie_ids = movies["movieId"].dropna().astype(int).tolist()
    preds = []
    for mid in movie_ids:
        try:
            est = model.predict(int(user_id), int(mid)).est
            preds.append((mid, est))
        except Exception:
            # ignore any weird IDs
            continue

    # sort by predicted rating (desc), take top_n
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]

    # build a nice table
    top_map = {m: s for m, s in top}
    out = movies[movies["movieId"].isin([m for m, _ in top])].copy()
    out["predicted_rating"] = out["movieId"].map(top_map)
    out = out.sort_values("predicted_rating", ascending=False).reset_index(drop=True)
    return out

# ---------- Action ----------
if go:
    with st.spinner("Thinking..."):
        results = recommend_movies(user_id=int(user_id), top_n=int(top_n))

    if results.empty:
        st.warning("No recomvenv\Scripts\activatemendations found. Try a different User ID.")
    else:
        st.subheader("Top Recommendations")
        st.dataframe(results)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", data=csv, file_name="recommendations.csv", mime="text/csv")

st.caption("Tip: If the user ID was not seen during training, the model may return near-average scores.")
