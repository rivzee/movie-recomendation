# 🎬 Anime Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

A smart, content-based anime recommendation engine powered by Machine Learning. This application helps users discover their next favorite anime by analyzing similarity across genres, types, studios, and ratings.

![Project Preview](app/static/hero-banner.png)
*(Replace with an actual screenshot of your app)*

## ✨ Features

- **🔍 Intelligent Search**: Uses fuzzy matching to find anime titles even with typos.
- **🤖 Hybrid Recommendation Engine**: Combines **K-Means Clustering** to narrow down candidates and **Cosine Similarity** to find the closest matches.
- **📊 Genre Explorer**: Enter a genre (e.g., "Action", "Romance") to see the top-rated anime in that category.
- **⚙️ Smart Filters**: Refine recommendations by **Type** (TV, Movie, OVA) and **Premiere Year**.
- **🖼️ Real-time Metadata**: Dynamically fetches the latest posters and details from MyAnimeList using web scraping.
- **📈 Data Visualization**: Generates similarity charts and score comparisons on the fly.

## 🛠️ Tech Stack

- **Web Framework**: Flask (Python)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (K-Means, Cosine Similarity, Scaler)
- **Visualization**: Matplotlib
- **Utilities**: RapidFuzz (String Matching), BeautifulSoup (Web Scraping)

## 📂 Project Structure

```bash
├── app/
│   ├── app.py                # Main Flask application entry point
│   ├── recommender.py        # Recommendation engine & chart generation logic
│   ├── static/               # CSS, Images, and generated charts
│   └── templates/            # HTML templates (Jinja2)
├── data/
│   └── dataset.csv           # Raw source dataset
├── models/
│   ├── anime_with_clusters.csv # Processed data with cluster labels
│   ├── kmeans.pkl            # Trained K-Means model
│   └── X_scaled.pkl          # Scaled feature matrix
├── notebooks/                # Jupyter Notebooks for analysis
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocess_train.ipynb # Data cleaning & Model training
│   └── ...
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Gioezzy/movie-recomendation.git
    cd movie-recomendation
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    cd app
    python app.py
    ```

4.  **Open in Browser**
    Navigate to `http://127.0.0.1:5000` to start using the recommender!

## 🧠 Data Science Workflow

This project follows a standard data science pipeline:

1.  **EDA (Exploratory Data Analysis)**: Analyzed distribution of genres, ratings, and types.
2.  **Preprocessing**: Cleaned missing values, encoded categorical features (One-Hot/Label Encoding), and scaled numerical data.
3.  **Clustering**: Applied **K-Means** to group similar anime into clusters, reducing the search space for recommendations.
4.  **Similarity**: Used **Cosine Similarity** within the identified cluster to rank the most relevant anime for the user.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📝 License

This project is licensed under the MIT License.
