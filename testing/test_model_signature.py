import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
# Set your remote tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow")


@pytest.mark.parametrize("model_name, alias, vectorizer_path", [
    ("ligbm_model_v1", "staging", "tfidf_vectorizer.pkl"), 
])
def test_model_with_vectorizer(model_name , alias ,vectorizer_path):
    client = MlflowClient()
    version_details = client.get_model_version_by_alias(model_name, alias)
    latest_version = version_details.version if version_details else None

    assert latest_version is not None , f"no model found in the {alias} state for {model_name}"
    try:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.lightgbm.load_model(model_uri)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
            
        # dummy input
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())
        # Predict using the model
        prediction = model.predict(input_df)
        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        # Verify the output shape (assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"
    
    except Exception as e:
        pytest.fail(f"Test Failed with Error  {e}")