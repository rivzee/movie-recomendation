import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv("../models/anime_with_clusters.csv")
X_scaled = joblib.load("../models/X_scaled.pkl")

titles = df["english"].dropna().tolist()

all_genres = set()
for g_str in df["genres"].dropna():
    for g in g_str.split(","):
        all_genres.add(g.strip())

# Helper to extract year
def extract_year(premiered_str):
    if pd.isna(premiered_str):
        return None
    import re
    match = re.search(r'\d{4}', str(premiered_str))
    if match:
        return int(match.group())
    return None

df['year'] = df['premiered'].apply(extract_year)

def fuzzy_match(query):
  matched = process.extractOne(query, titles)
  if matched and matched[1] >= 70:
    return matched[0]
  return None

def get_anime_image(mal_link):
    if pd.isna(mal_link):
        return None
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(mal_link, headers=headers, timeout=3)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img = soup.select_one('div.leftside img')
            if img:
                return img.get('data-src') or img.get('src')
    except Exception:
        pass
    return None

def recommend(query, anime_type=None, year=None, top_n=10):
    query_title = query.title() 
    matched_genre = None
    
    for g in all_genres:
        if query.lower() == g.lower():
            matched_genre = g
            break
            
    if matched_genre:
        mask = df["genres"].fillna("").str.contains(matched_genre, regex=False)
        genre_df = df[mask].copy()
        
        # Filter by Type
        if anime_type:
            genre_df = genre_df[genre_df["type"] == anime_type]
            
        # Filter by Year
        if year:
            genre_df = genre_df[genre_df["year"] == int(year)]
        
        genre_df["score"] = pd.to_numeric(genre_df["score"], errors="coerce")
        genre_df = genre_df.sort_values(by="score", ascending=False).head(top_n)
        
        results = []
        labels = []
        scores = []
        links = []
        
        rows = list(genre_df.iterrows())
        
        for i, row in rows:
            links.append(row["anime show link"])

        with ThreadPoolExecutor(max_workers=10) as executor:
             images = list(executor.map(get_anime_image, links))

        idx_counter = 0
        for i, row in rows:
            title = row["english"]
            if pd.isna(title):
                title = row["japanese"]
            
            # Helper for safe string
            studio = str(row["studios"]).replace("['", "").replace("']", "").replace("'", "")
            episodes = int(row["episodes"]) if pd.notna(row["episodes"]) else "?"
            status = row["status"] if pd.notna(row["status"]) else "Unknown"

            results.append({
                "title": title,
                "genres": row["genres"],
                "type": row["type"],
                "similarity": row["score"],
                "image": images[idx_counter],
                "episodes": f"{episodes} eps",
                "status": status,
                "studio": studio.split(",")[0], # Just take first studio if multiple
                "year": int(row["year"]) if pd.notna(row["year"]) else "",
                "premiered": row["premiered"] if pd.notna(row["premiered"]) else "Unknown",
                "mal_link": row["anime show link"]
            })
            labels.append(str(title)[:20])
            scores.append(row["score"])
            idx_counter += 1

        os.makedirs("static/charts", exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.barh(labels[::-1], scores[::-1], color='skyblue')
        plt.xlabel("Score (MyAnimeList)")
        plt.title(f"Top Rated {matched_genre} Anime")
        plt.tight_layout()
        plt.savefig("static/charts/genre_top.png", dpi=200)
        plt.close()
        
        return f"Top {matched_genre} Anime", results, "charts/genre_top.png"

    # 2. Existing Title-based Logic
    matched = fuzzy_match(query)
    if matched is None:
        return None, None, None
  
    idx = df[df["english"] == matched].index[0]
    cluster = df.loc[idx, "cluster"]
  
    cluster_idx = df[df["cluster"] == cluster].index
    
    # Filter by Type
    if anime_type:
        type_mask = df.loc[cluster_idx, "type"] == anime_type
        cluster_idx = cluster_idx[type_mask]
        
    # Filter by Year
    if year:
        year_mask = df.loc[cluster_idx, "year"] == int(year)
        cluster_idx = cluster_idx[year_mask]

    if len(cluster_idx) == 0:
         return matched, [], None # No results after filter
         
    sims = cosine_similarity([X_scaled[idx]], X_scaled[cluster_idx])[0]
  
    ranked = sorted(
        zip(cluster_idx, sims),
        key=lambda x: x[1],
        reverse=True
    )
  
    results = []
    labels = []
    scores = []
    links = []
    
    target_indices = ranked[1:top_n + 1]
    
    # Collect links first
    for i, s in target_indices:
        links.append(df.loc[i, "anime show link"])
        
    # Threaded Fetch
    with ThreadPoolExecutor(max_workers=10) as executor:
        images = list(executor.map(get_anime_image, links))
  
    idx_counter = 0
    for i, s in target_indices:
        title = df.loc[i, "english"]
        if pd.isna(title):
            title = df.loc[i, "japanese"]
            
        studio = str(df.loc[i, "studios"]).replace("['", "").replace("']", "").replace("'", "")
        episodes = int(df.loc[i, "episodes"]) if pd.notna(df.loc[i, "episodes"]) else "?"
        status = df.loc[i, "status"] if pd.notna(df.loc[i, "status"]) else "Unknown"

        results.append({
            "title": title,
            "genres": df.loc[i, "genres"],
            "type": df.loc[i, "type"],
            "similarity": round(float(s), 3),
            "image": images[idx_counter],
            "episodes": f"{episodes} eps",
            "status": status,
            "studio": studio.split(",")[0],
            "year": int(df.loc[i, "year"]) if pd.notna(df.loc[i, "year"]) else "",
            "premiered": df.loc[i, "premiered"] if pd.notna(df.loc[i, "premiered"]) else "Unknown",
            "mal_link": df.loc[i, "anime show link"]
        })
        labels.append(str(title)[:20])
        scores.append(s)
        idx_counter += 1
    
    os.makedirs("static/charts", exist_ok=True)
  
    plt.figure(figsize=(7, 4))
    plt.barh(labels[::-1], scores[::-1])
    plt.xlabel("Similarity Score")
    plt.title("Similarity with input Anime")
    plt.tight_layout()
    plt.savefig("static/charts/similarity.png", dpi=200)
    plt.close()
  
    return matched, results, "charts/similarity.png"