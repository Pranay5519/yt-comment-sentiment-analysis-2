FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy API
COPY fast_api/ /app/fast_api

# Copy only topic classification module
COPY src/comments_classification /app/src/comments_classification

# Model artifacts
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["uvicorn", "fast_api.app:app", "--host", "0.0.0.0", "--port", "8000"]