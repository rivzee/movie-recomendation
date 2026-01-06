import sys
import os

# Add app directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import the Flask app
from app.app import app

__all__ = ['app']
