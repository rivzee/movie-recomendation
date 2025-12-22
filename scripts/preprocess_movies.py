"""
Movie Recommendation System - Data Preprocessing Script
Processes MovieLens dataset to create features for recommendation
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib
import re
import os

print("=" * 60)
print("MOVIE RECOMMENDATION - DATA PREPROCESSING")
print("=" * 60)

# Paths - use script directory as base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# 1. LOAD MOVIES
# ============================================================
print("\n[1/7] Loading movies...")
movies = pd.read_csv(f"{DATA_DIR}/movie.csv")
print(f"    ✓ Loaded {len(movies)} movies")

# ============================================================
# 2. LOAD RATINGS (Aggregated to save memory)
# ============================================================
print("\n[2/7] Processing ratings (this may take a while)...")

# Process ratings in chunks to handle large file
chunk_size = 1_000_000
rating_agg = None

for i, chunk in enumerate(pd.read_csv(f"{DATA_DIR}/rating.csv", chunksize=chunk_size)):
    chunk_stats = chunk.groupby('movieId').agg(
        rating_sum=('rating', 'sum'),
        rating_count=('rating', 'count')
    )
    
    if rating_agg is None:
        rating_agg = chunk_stats
    else:
        rating_agg = rating_agg.add(chunk_stats, fill_value=0)
    
    print(f"    Processed chunk {i+1} ({(i+1)*chunk_size:,} ratings)")

# Calculate average
rating_agg['avg_rating'] = rating_agg['rating_sum'] / rating_agg['rating_count']
rating_agg = rating_agg.reset_index()
rating_agg = rating_agg[['movieId', 'avg_rating', 'rating_count']]

print(f"    ✓ Processed all ratings")

# ============================================================
# 3. EXTRACT YEAR FROM TITLE
# ============================================================
print("\n[3/7] Extracting year from titles...")

def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        return int(match.group(1))
    return None

def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).strip()

movies['year'] = movies['title'].apply(extract_year)
movies['clean_title'] = movies['title'].apply(clean_title)
print(f"    ✓ Year range: {movies['year'].min()} - {movies['year'].max()}")

# ============================================================
# 4. MERGE RATINGS WITH MOVIES
# ============================================================
print("\n[4/7] Merging ratings with movies...")
movies = movies.merge(rating_agg, on='movieId', how='left')
movies['avg_rating'] = movies['avg_rating'].fillna(0)
movies['rating_count'] = movies['rating_count'].fillna(0).astype(int)
print(f"    ✓ Movies with ratings: {(movies['rating_count'] > 0).sum()}")

# ============================================================
# 5. PROCESS GENRES
# ============================================================
print("\n[5/7] Processing genres...")

# Get all unique genres
all_genres = set()
for genres in movies['genres'].dropna():
    for g in genres.split('|'):
        if g != '(no genres listed)':
            all_genres.add(g)

genre_list = sorted(list(all_genres))
print(f"    ✓ Found {len(genre_list)} genres: {genre_list}")

# Create genre features
for genre in genre_list:
    movies[f'genre_{genre}'] = movies['genres'].fillna('').apply(
        lambda x: 1 if genre in x.split('|') else 0
    )

# ============================================================
# 6. CREATE FEATURE MATRIX & CLUSTERING
# ============================================================
print("\n[6/7] Creating feature matrix and clusters...")

genre_cols = [col for col in movies.columns if col.startswith('genre_')]

# Normalize numerical features
movies['year_normalized'] = movies['year'].fillna(movies['year'].median())
movies['rating_normalized'] = movies['avg_rating']

# Create feature matrix
feature_cols = genre_cols + ['rating_normalized', 'year_normalized']
X = movies[feature_cols].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"    ✓ Feature matrix shape: {X_scaled.shape}")

# Clustering
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
movies['cluster'] = kmeans.fit_predict(X_scaled)
print(f"    ✓ Created {n_clusters} clusters")

# ============================================================
# 7. SAVE PROCESSED DATA
# ============================================================
print("\n[7/7] Saving processed data...")

# Save processed movies
movies.to_csv(f'{MODELS_DIR}/movies_with_clusters.csv', index=False)
print(f"    ✓ Saved: movies_with_clusters.csv")

# Save scaled features
joblib.dump(X_scaled, f'{MODELS_DIR}/movies_X_scaled.pkl')
print(f"    ✓ Saved: movies_X_scaled.pkl")

# Save scaler
joblib.dump(scaler, f'{MODELS_DIR}/movies_scaler.pkl')
print(f"    ✓ Saved: movies_scaler.pkl")

# Save genre list
joblib.dump(genre_list, f'{MODELS_DIR}/genre_list.pkl')
print(f"    ✓ Saved: genre_list.pkl")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print(f"Total Movies: {len(movies)}")
print(f"Movies with ratings: {(movies['rating_count'] > 0).sum()}")
print(f"Year Range: {int(movies['year'].min())} - {int(movies['year'].max())}")
print(f"Genres: {len(genre_list)}")
print(f"Feature Dimensions: {X_scaled.shape[1]}")
print(f"Clusters: {n_clusters}")

# Top rated movies preview
print("\n📊 Top 10 Highest Rated Movies (1000+ ratings):")
popular = movies[movies['rating_count'] >= 1000].nlargest(10, 'avg_rating')
for i, row in popular.iterrows():
    print(f"   ⭐ {row['avg_rating']:.2f} | {row['clean_title']} ({int(row['year']) if pd.notna(row['year']) else 'N/A'})")

print("\n" + "=" * 60)
