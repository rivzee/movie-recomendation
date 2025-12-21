from flask import Flask, render_template, request
from recommender import recommend
import pandas as pd
import os

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

df = pd.read_csv("../models/anime_with_clusters.csv")
titles = sorted(df["english"].dropna().unique())


@app.route("/")
def index():
    return render_template("index.html", titles=titles)


@app.route("/recommend", methods=["POST"])
def recommend_view():
    query = request.form.get("title")
    anime_type = request.form.get("type")
    year = request.form.get("year")
    
    # Pass None if empty string
    if not anime_type:
        anime_type = None
    if not year:
        year = None

    matched, results, chart = recommend(query, anime_type=anime_type, year=year)

    if results is None:
        return render_template(
            "index.html",
            titles=titles,
            error="Judul tidak ditemukan, coba ejaan lain."
        )

    return render_template(
        "result.html",
        query=query,
        matched=matched,
        results=results,
        chart=chart
    )


if __name__ == "__main__":
    app.run(debug=True)
