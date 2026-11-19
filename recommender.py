import pandas as pd
import json
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Ngarkimi i dataset-it
movies = pd.read_csv("data/tmdb_5000_movies.csv")


# 2. Funksionet për nxjerrjen e emrave nga genres dhe keywords
def extract_names(text):
    """Version pa hapësira, për TF-IDF (ScienceFiction)."""
    try:
        items = json.loads(text)
        return " ".join(item["name"].replace(" ", "") for item in items)
    except (TypeError, ValueError, KeyError):
        return ""


def pretty_genres(text):
    """Version i bukur për t'u shfaqur (Science Fiction, Action)."""
    try:
        items = json.loads(text)
        return ", ".join(item["name"] for item in items)
    except (TypeError, ValueError, KeyError):
        return ""


# Fjalor për t'i kthyer fjalët-çelës në formë të lexueshme (spacetravel -> space travel)
_pretty_terms = {}
for _raw in movies["keywords"]:
    try:
        for _it in json.loads(_raw):
            _pretty_terms[_it["name"].replace(" ", "").lower()] = _it["name"]
    except (TypeError, ValueError, KeyError):
        pass


# 3. Përpunimi i kolonave
movies["genres_display"] = movies["genres"].apply(pretty_genres)
movies["genres"] = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)
movies["overview"] = movies["overview"].fillna("")


# 4. Përmbajtja e përbashkët për çdo film
movies["combined_features"] = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["overview"]
)


# 5. Modeli TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

_feature_names = tfidf.get_feature_names_out()
_genre_tokens = set()
for _g in movies["genres"]:
    _genre_tokens.update(t.lower() for t in _g.split())


def _readable(term):
    return _pretty_terms.get(term, term)


def _good_term(term):
    return term not in _genre_tokens and not term.isdigit() and len(term) >= 4


def _indices_for(titles):
    indices = []
    for title in titles:
        matches = movies.index[
            movies["title"].str.casefold() == title.strip().casefold()
        ]
        if len(matches) == 0:
            raise ValueError(f"Filmi '{title}' nuk u gjet në dataset.")
        indices.append(matches[0])
    return indices


# 6. Rekomandimet
def recommend_movies(selected_titles, top_n=5):

    if len(selected_titles) != 3:
        raise ValueError("Duhet të zgjidhen saktësisht 3 filma.")

    selected_indices = _indices_for(selected_titles)

    if len(set(selected_indices)) != 3:
        raise ValueError("Zgjidhni 3 filma të ndryshëm.")

    user_profile = np.asarray(tfidf_matrix[selected_indices].mean(axis=0))

    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    similarity_scores[selected_indices] = -1

    recommended_indices = similarity_scores.argsort()[::-1][:top_n]

    # genres_display riemërtohet "genres" që app.py ta marrë të bukur
    recommendations = movies.iloc[recommended_indices][
        ["title", "genres_display"]
    ].copy()
    recommendations = recommendations.rename(columns={"genres_display": "genres"})

    recommendations["similarity_score"] = similarity_scores[recommended_indices]

    return recommendations


# 7. Profili i shijes: çfarë kanë të përbashkët filmat e zgjedhur
def taste_profile(selected_titles, k=6):
    idx = _indices_for(selected_titles)

    profile = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
    themes = []
    for i in profile.argsort()[::-1]:
        term = _feature_names[i]
        if profile[i] <= 0:
            break
        if _good_term(term):
            themes.append(_readable(term))
        if len(themes) == k:
            break

    counts = Counter()
    for g in movies.loc[idx, "genres_display"]:
        for name in g.split(", "):
            if name:
                counts[name] += 1

    return {"genres": counts.most_common(5), "themes": themes}


# 8. Shpjegimi: pse përputhet një film me zgjedhjet e tua
def explain(selected_titles, rec_title, k=4):
    idx = _indices_for(selected_titles)
    rec_idx = _indices_for([rec_title])[0]

    profile = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
    rec_vec = tfidf_matrix[rec_idx].toarray().ravel()
    contribution = profile * rec_vec

    themes = []
    for i in contribution.argsort()[::-1]:
        if contribution[i] <= 0:
            break
        term = _feature_names[i]
        if _good_term(term):
            themes.append(_readable(term))
        if len(themes) == k:
            break

    picked = set()
    for g in movies.loc[idx, "genres_display"]:
        picked.update(x for x in g.split(", ") if x)
    rec_genres = [x for x in movies.loc[rec_idx, "genres_display"].split(", ") if x]
    shared = [g for g in rec_genres if g in picked]

    return {"genres": shared, "themes": themes}


# 9. Test
if __name__ == "__main__":
    favorite_movies = ["Inception", "Interstellar", "The Martian"]
    results = recommend_movies(favorite_movies)

    print("\nFilmat e preferuar:")
    for movie in favorite_movies:
        print("-", movie)

    print("\nProfili:", taste_profile(favorite_movies))
    print("\n5 rekomandimet e gjeneruara nga RECOMIND:")
    print(results.to_string(index=False))