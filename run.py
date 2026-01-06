import os
from app import app

if __name__ == "__main__":
    # Use PORT from environment for Render, default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    print(f"Starting Movie Recommendation System on port {port}...")
    app.run(debug=debug, host='0.0.0.0', port=port)

