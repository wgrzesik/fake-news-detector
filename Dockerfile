# ============================================================
# Dockerfile — FastAPI Prediction API
# ============================================================
# Builds a self-contained image with the 3 models the API uses.
# Usage:
#   docker build -t fake-news-api .
#   docker run -p 8000:8000 fake-news-api
# ============================================================

FROM python:3.11-slim

# Install system dependencies required by some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source code ───────────────────────────
COPY research/ ./research/
COPY backend/  ./backend/

# ── Copy ONLY the 3 models the API needs (~2.5 MB) ─────────
#    short_text  → LIAR    / lr  / word2vec
#    long_article→ ISOT    / svm / tfidf
#    general     → WELFake / xgb / tfidf
COPY saved_models/LIAR/lr_word2vec/    ./saved_models/LIAR/lr_word2vec/
COPY saved_models/ISOT/svm_tfidf/     ./saved_models/ISOT/svm_tfidf/
COPY saved_models/WELFake/xgb_tfidf/  ./saved_models/WELFake/xgb_tfidf/

EXPOSE 8000

# ── Start the FastAPI server ────────────────────────────────
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

