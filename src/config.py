"""CTIC Curriculum Engine configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Database — use individual fields to avoid URL-encoding issues with special chars in passwords
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Legacy: fallback to DATABASE_URL if individual fields not set
DATABASE_URL = os.getenv("DATABASE_URL", "")

# CTIC website
BASE_URL = "https://ctinventionconvention.org"
GRADE_BAND_PAGES = {
    "K-2": "/k-2-curriculum",
    "3-5": "/3-5-curriculum",
    "6-8": "/6-8-curriculum",
    "9-12": "/9-12-curriculum",
}

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Google Drive (optional service account)
GOOGLE_SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY", "")

# Gemini API (Part 2 — chat engine)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
