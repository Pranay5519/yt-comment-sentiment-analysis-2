import pytest
import requests
import os
BASE_URL = "http://127.0.0.1:8000"
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.integration
def test_topics_endpoint():

    payload = {
  "comments": [
            {
                "text": "This video editing is amazing",
                "timestamp": "2026-02-01T02:35:03Z",
                "authorId": "user1"
            },
            {
                "text": "Camera quality is insane",
                "timestamp": "2026-01-31T18:59:54Z",
                "authorId": "user2"
            }
        ],
  "api_key": os.getenv("GOOGLE_API_KEY"),
  "model_name": "gemini-2.5-flash",
  "temperature": 0
}

    url = f"{BASE_URL}/topics"

    response = requests.post(url, json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "topics" in data
    assert "classified_comments" in data