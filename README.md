# 🎬 Movie Recommendation System

Sistem rekomendasi film berbasis Machine Learning menggunakan dataset MovieLens (27,000+ film dengan 20 juta+ rating).

![Movie Recommender](https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=800)

## ✨ Fitur

- 🔍 **Pencarian Film** - Cari berdasarkan judul film (contoh: "Batman", "Star Wars", "Toy Story")
- 🎭 **Pencarian Genre** - Cari berdasarkan genre (contoh: "Action", "Comedy", "Drama")
- 📊 **Rekomendasi Cerdas** - Sistem menemukan film yang relevan berdasarkan pencarian
- 🖼️ **Poster Film** - Menampilkan poster dari TMDB API
- ⭐ **Rating & Statistik** - Menampilkan rating rata-rata dan jumlah votes
- 🔎 **Autocomplete** - Saran judul film saat mengetik
- 📈 **Visualisasi** - Grafik perbandingan rating/similarity
- 🌐 **REST API** - Endpoint API untuk integrasi

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
cd "d:\machine learning\movie-recomendation"
pip install -r requirements.txt
```

### 2. Jalankan Preprocessing (Opsional - hanya jika belum ada file model)

```bash
python scripts\preprocess_movies.py
```

> ⏱️ Proses ini memakan waktu ~3-5 menit karena memproses 20 juta rating

### 3. Jalankan Aplikasi

```bash
cd app
python app.py
```

### 4. Buka di Browser

Akses: **http://127.0.0.1:5000**

## 📁 Struktur Project

```
movie-recomendation/
├── app/
│   ├── app.py                  # Flask web server
│   ├── movie_recommender.py    # Core recommendation logic
│   ├── static/                 # CSS, images, charts
│   └── templates/              # HTML templates
│       ├── index.html          # Homepage
│       └── result.html         # Search results page
├── data/
│   ├── movie.csv               # Dataset film (27K films)
│   └── rating.csv              # Dataset rating (20M+ ratings)
├── models/
│   ├── movies_with_clusters.csv  # Processed movie data
│   ├── movies_X_scaled.pkl       # Feature matrix
│   ├── movies_scaler.pkl         # Scaler model
│   └── genre_list.pkl            # List of genres
├── scripts/
│   └── preprocess_movies.py    # Preprocessing script
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Dependencies
└── README.md                   # Dokumentasi ini
```

## 🎯 Cara Menggunakan

### Pencarian Judul Film
Ketik judul film untuk menemukan film tersebut beserta yang serupa:
- `Batman` → Semua film Batman
- `Star Wars` → Semua film Star Wars
- `The Matrix` → Film Matrix dan sequel-nya
- `Avengers` → Film Avengers
- `Toy Story` → Film Toy Story

### Pencarian Genre
Ketik nama genre untuk melihat film terbaik di genre tersebut:
- `Action` → Top film Action
- `Comedy` → Top film Comedy
- `Drama` → Top film Drama
- `Horror` → Top film Horror
- `Sci-Fi` → Top film Science Fiction
- `Romance` → Top film Romance
- `Animation` → Top film Animation
- `Thriller` → Top film Thriller

### Filter
Gunakan dropdown untuk filter tambahan:
- **Genre** - Filter berdasarkan genre spesifik
- **Tahun** - Filter berdasarkan tahun rilis

## 🔌 API Endpoints

| Endpoint | Method | Deskripsi | Contoh |
|----------|--------|-----------|--------|
| `/api/search` | GET | Autocomplete pencarian | `/api/search?q=batman&limit=5` |
| `/api/recommend` | GET | Rekomendasi film | `/api/recommend?q=Action` |
| `/api/popular` | GET | Film populer | `/api/popular?genre=Comedy&limit=10` |
| `/api/genres` | GET | Daftar genre | `/api/genres` |
| `/api/stats` | GET | Statistik database | `/api/stats` |

### Contoh Response API

```json
// GET /api/recommend?q=Batman
{
  "query": "Batman",
  "matched": "Hasil pencarian 'Batman'",
  "count": 10,
  "results": [
    {
      "title": "Batman",
      "year": 1989,
      "genres": "Action, Crime, Thriller",
      "rating": 3.4,
      "rating_count": 59184,
      "poster": "https://image.tmdb.org/t/p/w500/..."
    },
    ...
  ]
}
```

## 📊 Dataset

Dataset yang digunakan adalah **MovieLens** dari Kaggle:
- **27,278 film** dengan metadata (judul, genre, tahun)
- **20,000,264 rating** dari pengguna
- **20 genre** tersedia
- **Rentang tahun**: 1874 - 2019

## 🛠️ Teknologi

- **Backend**: Python, Flask
- **ML/Data**: Pandas, NumPy, Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualisasi**: Matplotlib
- **Fuzzy Match**: RapidFuzz
- **Poster**: TMDB API

## 📝 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
flask
jinja2
rapidfuzz
requests
beautifulsoup4
```

## 👤 Author

Movie Recommendation System - Machine Learning Project

---

Made with ❤️ using Python & Machine Learning
