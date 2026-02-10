FROM python:3.11-slim

WORKDIR /app

# Needed for lightgbm + some ML libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy only the root locked requirements
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy FastAPI source (flattened into /app)
COPY fast_api/ /app/

# Model / artifacts
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Only if you really use nltk
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

# Because app.py is now directly inside /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
