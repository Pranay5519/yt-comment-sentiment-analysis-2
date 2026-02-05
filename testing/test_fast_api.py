import pytest
import requests
import os
BASE_URL = "http://localhost:8000"


# @pytest.mark.integration
# def test_fetch_comments_endpoint():

#     payload = {
#     "video_id": "dQw4w9WgXcQ",
#     "api_key": NONE,
#     "max_comments": 4
#     }

#     url = f"{BASE_URL}/fetch_comments"
#     response = requests.post(url, json=payload)

#     assert response.status_code == 200

#     data = response.json()
#     assert isinstance(data, list)
#     assert len(data) > 0
#     assert isinstance(data[0], dict)


@pytest.mark.integration
def test_predict_endpoint():

    payload = {
        "comments": [
           {
            "text": "1st February 2026",
            "timestamp": "2026-02-01T02:35:03Z",
            "authorId": "UCG-deVShIq5Vlz6xAnu-c6Q"
        },
        {
            "text": "비버 목소리는 쵝오👍👍👍👍👍",
            "timestamp": "2026-01-31T18:59:54Z",
            "authorId": "UCqtvJz5bKooVFA661V4T_ow"
        }
        ]
    }

    url = f"{BASE_URL}/predict"
    response = requests.post(url, json=payload)

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.integration
def test_predict_with_timestamps_endpoint():

    payload = {
        "comments": [
           {
            "text": "1st February 2026",
            "timestamp": "2026-02-01T02:35:03Z",
            "authorId": "UCG-deVShIq5Vlz6xAnu-c6Q"
        },
        {
            "text": "비버 목소리는 쵝오👍👍👍👍👍",
            "timestamp": "2026-01-31T18:59:54Z",
            "authorId": "UCqtvJz5bKooVFA661V4T_ow"
        }
        ]
    }

    url = f"{BASE_URL}/predict_with_timestamps"
    response = requests.post(url, json=payload)

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert "sentiment" in data[0]


@pytest.mark.integration
def test_generate_chart_endpoint():

    payload = {
        "sentiment_counts": {
            "1": 4,
            "0": 2,
            "-1": 1
        }
    }

    url = f"{BASE_URL}/generate_chart"
    response = requests.post(url, json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 0


@pytest.mark.integration
def test_generate_wordcloud_endpoint():

    payload = {
        "comments": [
            {
    "text": "1st February 2026",
    "timestamp": "2026-02-01T02:35:03Z",
    "authorId": "UCG-deVShIq5Vlz6xAnu-c6Q"
  },
  {
    "text": "비버 목소리는 쵝오👍👍👍👍👍",
    "timestamp": "2026-01-31T18:59:54Z",
    "authorId": "UCqtvJz5bKooVFA661V4T_ow"
  }
        ]
    }

    url = f"{BASE_URL}/generate_wordcloud"
    response = requests.post(url, json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 0


@pytest.mark.integration
def test_generate_trend_graph_endpoint():

    payload = {
        "sentiment_data": [
            {"timestamp": "2024-10-01", "sentiment": 1},
            {"timestamp": "2024-10-02", "sentiment": 0},
            {"timestamp": "2024-10-03", "sentiment": -1}
        ]
    }

    url = f"{BASE_URL}/generate_trend_graph"
    response = requests.post(url, json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 0
