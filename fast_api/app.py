import matplotlib
matplotlib.use("Agg")
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import nltk
#nltk.download('stopwords')
from datetime import datetime   
app = FastAPI()
model = None
vectorizer = None
# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------
# Preprocessing
# -------------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {
            "not", "but", "however", "no", "yet"
        }

        comment = " ".join(
            [word for word in comment.split() if word not in stop_words]
        )

        lemmatizer = WordNetLemmatizer()
        comment = " ".join(
            [lemmatizer.lemmatize(word) for word in comment.split()]
        )

        return comment

    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# -------------------------
# MLflow loader
# -------------------------
def load_model_and_vectorizer(model_name, model_alias, vectorizer_path):
    mlflow.set_tracking_uri(
        "https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow"
    )

    client = MlflowClient()

    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.lightgbm.load_model(model_uri)

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


@app.on_event("startup")
def load_artifacts_once():
    global model, vectorizer

    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer(
            "light_gbm_model",
            "production",
            "./tfidf_vectorizer.pkl"
        )
# -------------------------
# Request schemas
# -------------------------
class Comment(BaseModel):
    text: str
    timestamp: datetime
    authorId: str

class PredictRequest(BaseModel):
    comments: List[Comment]
 
class PredictWithTimestampRequest(BaseModel):
    comments: List[Comment]


class ChartRequest(BaseModel):
    sentiment_counts: Dict[str, Any]


class WordcloudRequest(BaseModel):
    comments: List[Comment]


class TrendGraphRequest(BaseModel):
    sentiment_data: List[Dict[str, Any]] 
       
class FetchCommentsRequest(BaseModel):
    video_id: str
    api_key: str
    max_comments: int = 2000
# -------------------------
# Routes
# -------------------------

@app.get("/")
def home():
    return {"message": "Welcome to our FastAPI API"}

#--------------------------
# /fetch comments
#--------------------------
@app.post("/fetch_comments")
def fetch_comments_api(req: FetchCommentsRequest):

    comments = []
    page_token = ""

    try:
        while len(comments) < req.max_comments:

            url = "https://www.googleapis.com/youtube/v3/commentThreads"

            params = {
                "part": "snippet",
                "videoId": req.video_id,
                "maxResults": 100,
                "pageToken": page_token,
                "key": req.api_key
            }

            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "items" in data:
                for item in data["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]

                    comment_text = snippet.get("textOriginal")
                    timestamp = snippet.get("publishedAt")
                    author_id = (
                        snippet.get("authorChannelId", {}).get("value", "Unknown")
                    )

                    comments.append({
                        "text": comment_text,
                        "timestamp": timestamp,
                        "authorId": author_id
                    })

                    if len(comments) >= req.max_comments:
                        break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return comments

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching comments: {str(e)}"
        )

# -------------------------
# /predict
# -------------------------
@app.post("/predict")
def predict(comments: PredictRequest):
    
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        # Preprocess comments
        preprocessed_comments = [
            preprocess_comment(c.text) for c in comments.comments
        ]
        print("preprocessing Done")
        # Vectorize comments (sparse matrix)
        transformed_comments = vectorizer.transform(preprocessed_comments)
        print("Transformation Done")

        # Get expected schema columns from MLflow model
        #input_schema = model.metadata.get_input_schema()
        #expected_columns = input_schema.input_names()

        # Convert sparse matrix to DataFrame with vectorizer features
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame(
                    transformed_comments.toarray(),
                    columns=feature_names
                )
        print("Data Frame Generated")
# 🔥 correct alignment for MLflow
        #df = df.reindex(columns=expected_columns, fill_value=0.0)
        ## Reorder columns exactly as model expects
        #df = df[expected_columns]

        # Make predictions
        predictions = model.predict(df).tolist()
        # Convert predictions to strings
        predictions = [str(pred) for pred in predictions]
        print("predictions : ", predictions)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # Return response (same structure as Flask)
    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments.comments, predictions)
    ]

    return response # also return prediction for streamlit 
# -------------------------
# /predict_with_timestamps
# -------------------------
@app.post("/predict_with_timestamps")
def predict_with_timestamps(comments: PredictWithTimestampRequest):

    if not comments.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        texts = [item.text for item in comments.comments]
        timestamps = [item.timestamp for item in comments.comments]

        print("Extracted comments and timestamps")

        preprocessed_comments = [
            preprocess_comment(text) for text in texts
        ]
        print("Done preprocess comments")

        transformed_comments = vectorizer.transform(preprocessed_comments)
        print("Transformed comments")

        feature_names = vectorizer.get_feature_names_out()

        df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )

        predictions = model.predict(df).tolist()
        predictions = [str(p) for p in predictions]

        print("Done predictions:", predictions)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    response = [
        {
            "comment": c.text,
            "sentiment": s,
            "timestamp": t
        }
        for c, s, t in zip(comments.comments, predictions, timestamps)
    ]

    return response


# -------------------------
# /generate_chart
# -------------------------
@app.post("/generate_chart")
def generate_chart(req: ChartRequest):

    sentiment_counts = req.sentiment_counts

    if not sentiment_counts:
        raise HTTPException(
            status_code=400,
            detail="No sentiment counts provided"
        )

    try:
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]

        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chart generation failed: {str(e)}"
        )
# -------------------------
# /generate_wordcloud
# -------------------------
@app.post("/generate_wordcloud")
def generate_wordcloud(comments: WordcloudRequest):

    if not comments:
        raise HTTPException(
            status_code=400,
            detail="No comments provided"
        )

    try:
        preprocessed_comments = [
            preprocess_comment(c.text) for c in comments.comments
        ]

        text = " ".join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Word cloud generation failed: {str(e)}"
        )
# -------------------------
# /generate_trend_graph
# -------------------------
@app.post("/generate_trend_graph")
def generate_trend_graph(req: TrendGraphRequest):

    if not req.sentiment_data:
        raise HTTPException(
            status_code=400,
            detail="No sentiment data provided"
        )

    try:
        df = pd.DataFrame(req.sentiment_data)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        monthly_counts = (
            df.resample("ME")["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
        )

        monthly_totals = monthly_counts.sum(axis=1)

        monthly_percentages = (
            monthly_counts.T / monthly_totals
        ).T * 100

        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {
            -1: "red",
            0: "gray",
            1: "green",
        }

        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker="o",
                linestyle="-",
                label=sentiment_labels[val],
                color=colors[val],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )
        plt.gca().xaxis.set_major_locator(
            mdates.AutoDateLocator(maxticks=12)
        )

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trend graph generation failed: {str(e)}"
        )