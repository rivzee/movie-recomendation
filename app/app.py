"""
Movie Recommendation System - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify, redirect
from movie_recommender import (
    recommend, 
    get_popular_movies, 
    get_all_genres, 
    get_years,
    search_titles,
    get_movie_count,
    get_popular_titles,
    browse_movies,
    get_random_movies,
    get_movies_by_year,
    get_movie_detail,
    get_similar_movies,
    browse_movies_multi_genre
)
import os

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Get available genres and years for filters
GENRES = get_all_genres()
YEARS = get_years()
MOVIE_COUNT = get_movie_count()
POPULAR_TITLES = get_popular_titles(50)


@app.route("/")
def index():
    """Homepage with popular movies"""
    popular = get_popular_movies(limit=12)
    return render_template("index.html", 
                         popular_movies=popular,
                         genres=GENRES,
                         years=YEARS,
                         movie_count=MOVIE_COUNT,
                         popular_titles=POPULAR_TITLES[:20])


@app.route("/browse")
def browse():
    """Browse all movies with pagination - supports multi-genre"""
    # Get multiple genres from checkboxes
    genres_selected = request.args.getlist("genre")
    year = request.args.get("year", None)
    sort_by = request.args.get("sort", "rating_count")
    page = request.args.get("page", 1, type=int)
    
    # Use multi-genre browse if multiple genres selected
    if len(genres_selected) > 1:
        result = browse_movies_multi_genre(genres_selected, year=year, sort_by=sort_by, page=page, per_page=24)
        current_genres = genres_selected
    elif len(genres_selected) == 1:
        result = browse_movies(genre=genres_selected[0], year=year, sort_by=sort_by, page=page, per_page=24)
        current_genres = genres_selected
    else:
        result = browse_movies(genre=None, year=year, sort_by=sort_by, page=page, per_page=24)
        current_genres = []
    
    return render_template("browse.html",
                         movies=result["movies"],
                         total=result["total"],
                         page=result["page"],
                         total_pages=result["total_pages"],
                         genres=GENRES,
                         years=YEARS,
                         current_genres=current_genres,
                         current_year=year,
                         current_sort=sort_by)


@app.route("/recommend", methods=["POST"])
def recommend_view():
    """Handle recommendation requests - supports multi-genre and genre-only search"""
    query = request.form.get("title", "").strip()
    genres_selected = request.form.getlist("genre")  # Multiple genres
    year_filter = request.form.get("year", "")
    
    # If no title but genres selected, redirect to browse page
    if not query and genres_selected:
        genre_params = "&".join([f"genre={g}" for g in genres_selected])
        year_param = f"&year={year_filter}" if year_filter else ""
        return redirect(f"/browse?{genre_params}{year_param}")
    
    # If no title and no genres, show error
    if not query:
        return render_template("index.html",
                             popular_movies=get_popular_movies(limit=12),
                             genres=GENRES,
                             years=YEARS,
                             movie_count=MOVIE_COUNT,
                             popular_titles=POPULAR_TITLES[:20],
                             error="Silakan masukkan judul film atau pilih genre.")
    
    # Use first genre if multiple selected (or None)
    genre_filter = genres_selected[0] if genres_selected else None
    year_filter = year_filter if year_filter else None
    
    matched, results, chart = recommend(query, 
                                        genre_filter=genre_filter, 
                                        year_filter=year_filter,
                                        top_n=30)
    
    if results is None:
        return render_template("index.html",
                             popular_movies=get_popular_movies(limit=12),
                             genres=GENRES,
                             years=YEARS,
                             movie_count=MOVIE_COUNT,
                             popular_titles=POPULAR_TITLES[:20],
                             error=f"Film '{query}' tidak ditemukan. Coba judul atau genre lain.")
    
    if len(results) == 0:
        return render_template("result.html",
                             query=query,
                             matched=matched,
                             results=[],
                             chart=None,
                             genres=GENRES,
                             years=YEARS,
                             message="Tidak ada hasil dengan filter yang dipilih.")
    
    return render_template("result.html",
                         query=query,
                         matched=matched,
                         results=results,
                         chart=chart,
                         genres=GENRES,
                         years=YEARS)


@app.route("/api/search")
def api_search():
    """API endpoint for autocomplete search"""
    query = request.args.get("q", "")
    limit = request.args.get("limit", 10, type=int)
    
    if len(query) < 2:
        return jsonify([])
    
    results = search_titles(query, limit=limit)
    return jsonify(results)


@app.route("/api/recommend")
def api_recommend():
    """API endpoint for recommendations"""
    query = request.args.get("q", "")
    genre = request.args.get("genre", None)
    year = request.args.get("year", None)
    
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    matched, results, chart = recommend(query, 
                                        genre_filter=genre, 
                                        year_filter=year,
                                        top_n=30)
    
    if results is None:
        return jsonify({"error": f"Movie '{query}' not found"}), 404
    
    return jsonify({
        "query": query,
        "matched": matched,
        "results": results,
        "count": len(results),
        "chart": chart
    })


@app.route("/api/browse")
def api_browse():
    """API endpoint for browsing movies"""
    genre = request.args.get("genre", None)
    year = request.args.get("year", None)
    sort_by = request.args.get("sort", "rating_count")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    
    result = browse_movies(genre=genre, year=year, sort_by=sort_by, page=page, per_page=per_page)
    return jsonify(result)


@app.route("/api/popular")
def api_popular():
    """API endpoint for popular movies"""
    genre = request.args.get("genre", None)
    year = request.args.get("year", None)
    limit = request.args.get("limit", 20, type=int)
    
    results = get_popular_movies(genre=genre, year=year, limit=limit)
    return jsonify({"movies": results, "count": len(results)})


@app.route("/api/random")
def api_random():
    """API endpoint for random movie discovery"""
    count = request.args.get("count", 12, type=int)
    results = get_random_movies(count=count)
    return jsonify({"movies": results})


@app.route("/api/genres")
def api_genres():
    """API endpoint for available genres"""
    return jsonify({"genres": GENRES})


@app.route("/api/years")
def api_years():
    """API endpoint for available years"""
    return jsonify({"years": YEARS})


@app.route("/api/stats")
def api_stats():
    """API endpoint for movie database stats"""
    return jsonify({
        "total_movies": MOVIE_COUNT,
        "genres": len(GENRES),
        "genre_list": GENRES,
        "year_range": f"{min(YEARS)}-{max(YEARS)}"
    })


@app.route("/api/movie/<path:title>")
def api_movie_detail(title):
    """API endpoint for movie detail with trailer"""
    detail = get_movie_detail(title)
    if detail is None:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(detail)


@app.route("/api/similar/<path:title>")
def api_similar_movies(title):
    """API endpoint for similar movies"""
    limit = request.args.get("limit", 8, type=int)
    similar = get_similar_movies(title, limit=limit)
    return jsonify({"movie": title, "similar": similar})


@app.route("/api/browse/multi")
def api_browse_multi_genre():
    """API endpoint for browsing with multiple genres"""
    genres = request.args.getlist("genre")  # Multiple genre parameters
    year = request.args.get("year", None)
    sort_by = request.args.get("sort", "rating_count")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    
    if not genres:
        return jsonify({"error": "At least one genre is required"}), 400
    
    result = browse_movies_multi_genre(genres, year=year, sort_by=sort_by, page=page, per_page=per_page)
    return jsonify(result)


@app.route("/movie/<path:title>")
def movie_detail_page(title):
    """Movie detail page with trailer and similar movies"""
    detail = get_movie_detail(title)
    if detail is None:
        return render_template("index.html",
                             popular_movies=get_popular_movies(limit=12),
                             genres=GENRES,
                             years=YEARS,
                             movie_count=MOVIE_COUNT,
                             popular_titles=POPULAR_TITLES[:20],
                             error=f"Film '{title}' tidak ditemukan.")
    
    similar = get_similar_movies(title, limit=8)
    
    return render_template("movie_detail.html",
                         movie=detail,
                         similar_movies=similar,
                         genres=GENRES,
                         years=YEARS)


if __name__ == "__main__":
    print("🎬 Starting Movie Recommendation Server...")
    print(f"📊 Loaded {MOVIE_COUNT} movies with {len(GENRES)} genres")
    print("📍 Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
