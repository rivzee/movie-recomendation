"""
Movie Recommendation System
Content-based filtering using genres and ratings
"""

import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import os
import requests

# Configure professional chart style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#4a4a6a',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'grid.color': '#3a3a5a',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Load data
print("Loading movie recommendation data...")
df = pd.read_csv(os.path.join(BASE_DIR, "models", "movies_with_clusters.csv"))
X_scaled = joblib.load(os.path.join(BASE_DIR, "models", "movies_X_scaled.pkl"))
genre_list = joblib.load(os.path.join(BASE_DIR, "models", "genre_list.pkl"))

# Create title list for fuzzy matching
titles = df["clean_title"].dropna().tolist()

print(f"Loaded {len(df)} movies with {len(genre_list)} genres")

# Cache for poster URLs to avoid repeated API calls
_poster_cache = {}


def get_tmdb_poster(movie_title, year=None, use_cache=True):
    """Get movie poster from TMDB API with caching"""
    # Create cache key
    cache_key = f"{movie_title}_{year}"
    
    # Check cache first
    if use_cache and cache_key in _poster_cache:
        return _poster_cache[cache_key]
    
    try:
        api_key = "15d2ea6d0dc1d476efbca3eba2b9bbfb"
        
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": movie_title,
            "year": year if year else None
        }
        
        # Reduced timeout for faster response
        response = requests.get(url, params=params, timeout=2)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w300{poster_path}"
                    _poster_cache[cache_key] = poster_url
                    return poster_url
    except:
        pass
    
    # Cache the None result too to avoid retrying
    _poster_cache[cache_key] = None
    return None


def get_tmdb_poster_batch(movies, max_posters=6):
    """Get posters for a batch of movies, limiting API calls"""
    for i, movie in enumerate(movies):
        if i < max_posters:
            movie["poster"] = get_tmdb_poster(movie.get("title"), movie.get("year"))
        else:
            movie["poster"] = None
    return movies


def fuzzy_match(query):
    """Find closest movie title match with multiple strategies"""
    query_lower = query.lower().strip()
    
    # Strategy 1: Exact match (case insensitive)
    for title in titles:
        if title.lower() == query_lower:
            return title
    
    # Strategy 2: Partial match - query is contained in title
    partial_matches = []
    for title in titles:
        if query_lower in title.lower():
            partial_matches.append(title)
    
    if partial_matches:
        # Return shortest match (most specific)
        return min(partial_matches, key=len)
    
    # Strategy 3: Fuzzy match with low threshold
    matched = process.extractOne(query, titles, score_cutoff=40)
    if matched:
        return matched[0]
    
    return None


def create_professional_chart(labels, scores, title, chart_type="rating", filename="chart.png"):
    """Create a professional-looking chart with modern styling"""
    charts_dir = os.path.join(SCRIPT_DIR, "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    if not labels or not scores:
        return None
    
    # Reverse for better visual (highest at top)
    labels_rev = labels[::-1]
    scores_rev = scores[::-1]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels_rev) * 0.5)))
    
    # Create gradient colors based on scores
    if chart_type == "rating":
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(scores_rev)))
        xlabel = "Average Rating ⭐"
    else:
        colors = plt.cm.cool(np.linspace(0.2, 0.8, len(scores_rev)))
        xlabel = "Similarity Score 🎯"
    
    # Create horizontal bars
    bars = ax.barh(range(len(labels_rev)), scores_rev, color=colors, edgecolor='none', height=0.7)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores_rev)):
        width = bar.get_width()
        if chart_type == "rating":
            label_text = f'{score:.2f}'
        else:
            label_text = f'{score*100:.1f}%'
        
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                label_text, va='center', ha='left', 
                fontsize=10, fontweight='bold', color='#ffd700')
    
    # Customize axes
    ax.set_yticks(range(len(labels_rev)))
    ax.set_yticklabels(labels_rev, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=10)
    
    # Set title with emoji
    ax.set_title(f"📊 {title}", fontsize=16, fontweight='bold', pad=20, color='#ffffff')
    
    # Add subtle grid
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4a4a6a')
    ax.spines['bottom'].set_color('#4a4a6a')
    
    # Adjust x-axis limit
    max_score = max(scores_rev) if scores_rev else 1
    ax.set_xlim(0, max_score * 1.15)
    
    plt.tight_layout()
    
    filepath = os.path.join(charts_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    
    return f"charts/{filename}"


def recommend(query, genre_filter=None, year_filter=None, min_ratings=0, top_n=10):
    """
    Get movie recommendations based on query
    
    Args:
        query: Movie title or genre name
        genre_filter: Filter by specific genre
        year_filter: Filter by specific year
        min_ratings: Minimum number of ratings required
        top_n: Number of recommendations to return
    """
    query_lower = query.lower().strip()
    matched_genre = None
    
    # Check if query is a genre
    for g in genre_list:
        if query_lower == g.lower():
            matched_genre = g
            break
    
    # === GENRE-BASED RECOMMENDATION ===
    if matched_genre:
        mask = df[f"genre_{matched_genre}"] == 1
        genre_df = df[mask].copy()
        
        # Apply filters
        if genre_filter and genre_filter != matched_genre:
            if f"genre_{genre_filter}" in genre_df.columns:
                genre_df = genre_df[genre_df[f"genre_{genre_filter}"] == 1]
        
        if year_filter:
            genre_df = genre_df[genre_df["year"] == int(year_filter)]
            
        # Filter by minimum ratings
        genre_df = genre_df[genre_df["rating_count"] >= min_ratings]
        
        # Sort by rating
        genre_df = genre_df.sort_values("avg_rating", ascending=False).head(top_n)
        
        results = []
        labels = []
        scores = []
        
        for _, row in genre_df.iterrows():
            poster = get_tmdb_poster(row["clean_title"], row.get("year"))
            
            results.append({
                "title": row["clean_title"],
                "year": int(row["year"]) if pd.notna(row["year"]) else "",
                "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
                "rating": round(row["avg_rating"], 2),
                "rating_count": int(row["rating_count"]),
                "similarity": round(row["avg_rating"], 2),
                "poster": poster
            })
            labels.append(row["clean_title"][:30])
            scores.append(row["avg_rating"])
        
        # Create professional chart
        chart = create_professional_chart(
            labels, scores, 
            f"Top Rated {matched_genre} Movies", 
            chart_type="rating", 
            filename="genre_top.png"
        )
        
        return f"Top {matched_genre} Movies", results, chart
    
    # === TITLE-BASED SEARCH ===
    # First, find all movies that CONTAIN the search query
    matching_movies = df[df["clean_title"].str.lower().str.contains(query_lower, na=False)].copy()
    
    # Apply filters
    if genre_filter:
        if f"genre_{genre_filter}" in matching_movies.columns:
            matching_movies = matching_movies[matching_movies[f"genre_{genre_filter}"] == 1]
    
    if year_filter:
        matching_movies = matching_movies[matching_movies["year"] == int(year_filter)]
    
    # If we found movies containing the query, return those sorted by rating
    if len(matching_movies) > 0:
        # Sort by rating count and avg rating
        matching_movies = matching_movies.sort_values(
            ["rating_count", "avg_rating"], 
            ascending=[False, False]
        ).head(top_n)
        
        results = []
        labels = []
        scores = []
        
        for i, (_, row) in enumerate(matching_movies.iterrows()):
            # Only fetch posters for first 10 results
            poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < 10 else None
            
            results.append({
                "title": row["clean_title"],
                "year": int(row["year"]) if pd.notna(row["year"]) else "",
                "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
                "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
                "rating_count": int(row["rating_count"]),
                "similarity": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else 0,
                "poster": poster
            })
            labels.append(row["clean_title"][:30])
            scores.append(row["avg_rating"] if row["avg_rating"] > 0 else 0)
        
        # Create professional chart
        chart = create_professional_chart(
            labels, scores, 
            f"Movies matching '{query}'", 
            chart_type="rating", 
            filename="search_results.png"
        )
        
        return f"Hasil pencarian '{query}'", results, chart
    
    # === FUZZY SEARCH if no direct match ===
    matched = fuzzy_match(query)
    if matched is None:
        return None, None, None
    
    # Find the matched movie and get similar ones from same cluster
    idx = df[df["clean_title"] == matched].index[0]
    cluster = df.loc[idx, "cluster"]
    
    # Get movies in same cluster
    cluster_df = df[df["cluster"] == cluster].copy()
    
    # Apply filters
    if genre_filter:
        if f"genre_{genre_filter}" in cluster_df.columns:
            cluster_df = cluster_df[cluster_df[f"genre_{genre_filter}"] == 1]
    
    if year_filter:
        cluster_df = cluster_df[cluster_df["year"] == int(year_filter)]
    
    # Filter by minimum ratings
    cluster_df = cluster_df[cluster_df["rating_count"] >= min_ratings]
    
    if len(cluster_df) == 0:
        return matched, [], None
    
    cluster_idx = cluster_df.index
    
    # Calculate similarity
    sims = cosine_similarity([X_scaled[idx]], X_scaled[cluster_idx])[0]
    
    # Rank by similarity
    ranked = sorted(
        zip(cluster_idx, sims),
        key=lambda x: x[1],
        reverse=True
    )
    
    results = []
    labels = []
    scores = []
    
    # Skip first result (itself) and get top_n
    for i, s in ranked[1:top_n + 1]:
        row = df.loc[i]
        poster = get_tmdb_poster(row["clean_title"], row.get("year"))
        
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2),
            "rating_count": int(row["rating_count"]),
            "similarity": round(float(s), 3),
            "poster": poster
        })
        labels.append(row["clean_title"][:30])
        scores.append(s)
    
    # Create professional chart
    chart = create_professional_chart(
        labels, scores, 
        f"Movies Similar to '{matched}'", 
        chart_type="similarity", 
        filename="similarity.png"
    )
    
    return matched, results, chart


def get_popular_movies(genre=None, year=None, limit=20, max_posters=6):
    """Get popular movies for homepage - optimized for speed"""
    filtered = df[df["rating_count"] >= 1000].copy()
    
    if genre and f"genre_{genre}" in filtered.columns:
        filtered = filtered[filtered[f"genre_{genre}"] == 1]
    
    if year:
        filtered = filtered[filtered["year"] == int(year)]
    
    top_movies = filtered.nlargest(limit, "avg_rating")
    
    results = []
    for i, (_, row) in enumerate(top_movies.iterrows()):
        # Only fetch posters for first few movies to speed up loading
        poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < max_posters else None
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2),
            "rating_count": int(row["rating_count"]),
            "poster": poster
        })
    
    return results


def get_all_genres():
    """Get list of all genres"""
    return genre_list


def get_years():
    """Get list of available years"""
    years = df["year"].dropna().unique()
    return sorted([int(y) for y in years], reverse=True)


def search_titles(query, limit=10):
    """Search for movie titles matching query (for autocomplete)"""
    if not query or len(query) < 2:
        return []
    
    query_lower = query.lower().strip()
    
    # Find movies containing the query
    matches = df[df["clean_title"].str.lower().str.contains(query_lower, na=False)].copy()
    
    # Sort by rating count (popularity)
    matches = matches.sort_values("rating_count", ascending=False).head(limit)
    
    return [
        {
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else None
        }
        for _, row in matches.iterrows()
    ]


def get_movie_count():
    """Get total movie count"""
    return len(df)


def get_popular_titles(limit=100):
    """Get list of popular movie titles for suggestions"""
    popular = df[df["rating_count"] >= 1000].nlargest(limit, "rating_count")
    return popular["clean_title"].tolist()


def browse_movies(genre=None, year=None, sort_by="rating_count", page=1, per_page=20):
    """
    Browse all movies with filtering and pagination
    """
    filtered = df.copy()
    
    # Apply genre filter
    if genre and f"genre_{genre}" in filtered.columns:
        filtered = filtered[filtered[f"genre_{genre}"] == 1]
    
    # Apply year filter
    if year:
        filtered = filtered[filtered["year"] == int(year)]
    
    # Sort
    if sort_by == "rating":
        filtered = filtered.sort_values("avg_rating", ascending=False)
    elif sort_by == "year":
        filtered = filtered.sort_values("year", ascending=False)
    elif sort_by == "title":
        filtered = filtered.sort_values("clean_title")
    else:  # Default: rating_count (popularity)
        filtered = filtered.sort_values("rating_count", ascending=False)
    
    # Pagination
    total = len(filtered)
    total_pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    
    page_data = filtered.iloc[start:end]
    
    results = []
    for i, (_, row) in enumerate(page_data.iterrows()):
        # Only fetch posters for first few movies per page
        poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < 8 else None
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
            "rating_count": int(row["rating_count"]),
            "poster": poster
        })
    
    return {
        "movies": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }


def get_random_movies(count=12):
    """Get random movies for discovery"""
    sample = df[df["rating_count"] >= 100].sample(n=min(count, len(df)))
    
    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < 4 else None
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
            "rating_count": int(row["rating_count"]),
            "poster": poster
        })
    
    return results


def get_movies_by_year(year, limit=20):
    """Get top movies from a specific year"""
    year_movies = df[df["year"] == int(year)].copy()
    year_movies = year_movies.sort_values("rating_count", ascending=False).head(limit)
    
    results = []
    for i, (_, row) in enumerate(year_movies.iterrows()):
        poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < 6 else None
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
            "rating_count": int(row["rating_count"]),
            "poster": poster
        })
    
    return results


def get_movie_trailer(movie_title, year=None):
    """Get YouTube trailer key from TMDB API"""
    try:
        api_key = "15d2ea6d0dc1d476efbca3eba2b9bbfb"
        
        # First search for the movie to get its ID
        search_url = "https://api.themoviedb.org/3/search/movie"
        search_params = {
            "api_key": api_key,
            "query": movie_title,
            "year": year if year else None
        }
        
        response = requests.get(search_url, params=search_params, timeout=3)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                movie_id = results[0]["id"]
                
                # Get videos for this movie
                videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
                videos_params = {"api_key": api_key}
                
                vid_response = requests.get(videos_url, params=videos_params, timeout=3)
                if vid_response.status_code == 200:
                    videos = vid_response.json().get("results", [])
                    # Find YouTube trailer
                    for video in videos:
                        if video.get("site") == "YouTube" and video.get("type") in ["Trailer", "Teaser"]:
                            return video.get("key")  # YouTube video ID
    except:
        pass
    
    return None


def get_movie_overview(movie_title, year=None):
    """Get movie overview/synopsis from TMDB API in Indonesian"""
    try:
        api_key = "15d2ea6d0dc1d476efbca3eba2b9bbfb"
        
        # First search for the movie
        search_url = "https://api.themoviedb.org/3/search/movie"
        search_params = {
            "api_key": api_key,
            "query": movie_title,
            "year": year if year else None
        }
        
        response = requests.get(search_url, params=search_params, timeout=3)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                movie_id = results[0]["id"]
                
                # Get movie details in Indonesian
                detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                detail_params = {
                    "api_key": api_key,
                    "language": "id-ID"  # Indonesian
                }
                
                detail_response = requests.get(detail_url, params=detail_params, timeout=3)
                if detail_response.status_code == 200:
                    detail = detail_response.json()
                    overview = detail.get("overview", "")
                    
                    # If Indonesian not available, fallback to English
                    if not overview:
                        detail_params["language"] = "en-US"
                        detail_response = requests.get(detail_url, params=detail_params, timeout=3)
                        if detail_response.status_code == 200:
                            detail = detail_response.json()
                            overview = detail.get("overview", "")
                    
                    return overview if overview else "Sinopsis tidak tersedia."
    except:
        pass
    
    return "Sinopsis tidak tersedia."


# Cache for complete movie details
_movie_detail_cache = {}

def get_movie_detail(movie_title):
    """Get complete movie detail - OPTIMIZED with single search and caching"""
    # Check cache first
    cache_key = movie_title.lower()
    if cache_key in _movie_detail_cache:
        return _movie_detail_cache[cache_key]
    
    # Find the movie in our database
    movie_row = df[df["clean_title"].str.lower() == movie_title.lower()]
    
    if len(movie_row) == 0:
        match = fuzzy_match(movie_title)
        if match:
            movie_row = df[df["clean_title"] == match]
    
    if len(movie_row) == 0:
        return None
    
    row = movie_row.iloc[0]
    year = int(row["year"]) if pd.notna(row["year"]) else None
    title = row["clean_title"]
    
    # Initialize defaults
    poster = None
    trailer_key = None
    overview = "Sinopsis tidak tersedia."
    
    try:
        api_key = "15d2ea6d0dc1d476efbca3eba2b9bbfb"
        
        # Single search to get movie ID
        search_url = "https://api.themoviedb.org/3/search/movie"
        search_params = {"api_key": api_key, "query": title, "year": year}
        
        response = requests.get(search_url, params=search_params, timeout=2)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                movie_data = results[0]
                movie_id = movie_data["id"]
                
                # Get poster from search result
                if movie_data.get("poster_path"):
                    poster = f"https://image.tmdb.org/t/p/w300{movie_data['poster_path']}"
                
                # Get Indonesian overview (single call)
                detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                detail_response = requests.get(detail_url, params={"api_key": api_key, "language": "id-ID"}, timeout=2)
                if detail_response.status_code == 200:
                    detail = detail_response.json()
                    overview = detail.get("overview", "") or "Sinopsis tidak tersedia."
                    if overview == "Sinopsis tidak tersedia.":
                        # Fallback to English
                        overview = movie_data.get("overview", "") or "Sinopsis tidak tersedia."
                
                # Get trailer (single call)
                videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
                vid_response = requests.get(videos_url, params={"api_key": api_key}, timeout=2)
                if vid_response.status_code == 200:
                    videos = vid_response.json().get("results", [])
                    for video in videos:
                        if video.get("site") == "YouTube" and video.get("type") in ["Trailer", "Teaser"]:
                            trailer_key = video.get("key")
                            break
    except:
        pass
    
    result = {
        "title": title,
        "year": year or "",
        "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
        "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
        "rating_count": int(row["rating_count"]),
        "poster": poster,
        "trailer_key": trailer_key,
        "trailer_url": f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None,
        "overview": overview
    }
    
    # Cache the result
    _movie_detail_cache[cache_key] = result
    return result


def get_similar_movies(movie_title, limit=8):
    """Get similar movies based on genres and features"""
    # Find the movie
    movie_row = df[df["clean_title"].str.lower() == movie_title.lower()]
    
    if len(movie_row) == 0:
        match = fuzzy_match(movie_title)
        if match:
            movie_row = df[df["clean_title"] == match]
    
    if len(movie_row) == 0:
        return []
    
    idx = movie_row.index[0]
    movie_genres = movie_row.iloc[0]["genres"]
    
    if pd.isna(movie_genres):
        return []
    
    # Get genres of this movie
    genres = movie_genres.split("|")
    
    # Find movies with similar genres
    similar = df.copy()
    similar = similar[similar.index != idx]  # Exclude the movie itself
    
    # Score based on genre overlap
    similar["genre_score"] = 0
    for genre in genres:
        col = f"genre_{genre}"
        if col in similar.columns:
            similar["genre_score"] += similar[col]
    
    # Filter movies with at least one matching genre and good ratings
    similar = similar[(similar["genre_score"] > 0) & (similar["rating_count"] >= 100)]
    
    # Sort by genre score and rating
    similar = similar.sort_values(
        ["genre_score", "avg_rating", "rating_count"], 
        ascending=[False, False, False]
    ).head(limit)
    
    results = []
    for i, (_, row) in enumerate(similar.iterrows()):
        poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < 4 else None
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
            "rating_count": int(row["rating_count"]),
            "poster": poster
        })
    
    return results


def browse_movies_multi_genre(genres_list, year=None, sort_by="rating_count", page=1, per_page=20):
    """
    Browse movies with multiple genre filter (AND logic)
    """
    filtered = df.copy()
    
    # Apply multiple genre filters (movie must have ALL selected genres)
    for genre in genres_list:
        col = f"genre_{genre}"
        if col in filtered.columns:
            filtered = filtered[filtered[col] == 1]
    
    # Apply year filter
    if year:
        filtered = filtered[filtered["year"] == int(year)]
    
    # Sort
    if sort_by == "rating":
        filtered = filtered.sort_values("avg_rating", ascending=False)
    elif sort_by == "year":
        filtered = filtered.sort_values("year", ascending=False)
    elif sort_by == "title":
        filtered = filtered.sort_values("clean_title")
    else:  # Default: rating_count (popularity)
        filtered = filtered.sort_values("rating_count", ascending=False)
    
    # Pagination
    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    
    page_data = filtered.iloc[start:end]
    
    results = []
    for i, (_, row) in enumerate(page_data.iterrows()):
        poster = get_tmdb_poster(row["clean_title"], row.get("year")) if i < 8 else None
        results.append({
            "title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else "",
            "genres": row["genres"].replace("|", ", ") if pd.notna(row["genres"]) else "",
            "rating": round(row["avg_rating"], 2) if row["avg_rating"] > 0 else "N/A",
            "rating_count": int(row["rating_count"]),
            "poster": poster
        })
    
    return {
        "movies": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }


# Test
if __name__ == "__main__":
    print("\n--- Testing Movie Recommender ---")
    
    # Test genre search
    print("\nSearching for 'Action' movies:")
    title, results, chart = recommend("Action", top_n=5)
    if results:
        for r in results:
            print(f"  - {r['title']} ({r['year']}) - Rating: {r['rating']}")
    
    # Test title search
    print("\nSearching for movies similar to 'Toy Story':")
    title, results, chart = recommend("Toy Story", top_n=5)
    if results:
        for r in results:
            print(f"  - {r['title']} ({r['year']}) - Similarity: {r['similarity']}")
