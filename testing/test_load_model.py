import mlflow
import pytest
import os
from mlflow.tracking import MlflowClient

# Ensure authentication is set before anything else
# These must be set in your CI environment (GitHub Actions) or local terminal
# os.environ['MLFLOW_TRACKING_USERNAME'] = "Pranay5519"
# os.environ['MLFLOW_TRACKING_PASSWORD'] = "YOUR_TOKEN"

mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow")

@pytest.mark.parametrize("model_name, alias", [
    ("light_gbm_model", "staging"),
])
def test_load_latest_staging_model(model_name, alias):
    client = MlflowClient()
    
    # 1. Check if the alias exists and get the actual underlying URI
    try:
        version_details = client.get_model_version_by_alias(model_name, alias)
        # Using the direct source URI can sometimes bypass path resolution errors
        model_uri = f"models:/{model_name}@{alias}"
        
        # 2. Use pyfunc for better compatibility with remote storage
        model = mlflow.lightgbm.load_model(model_uri)

        assert model is not None
        print(f"Version {version_details.version} loaded successfully.")

    except Exception as e:
        # Instead of generic error, print the source to debug
        # Check if the 'source' path in version_details actually looks correct
        try:
            source = client.get_model_version_by_alias(model_name, alias).source
            print(f"Debug: Model source points to: {source}")
        except:
            pass
        pytest.fail(f"Model loading failed. Ensure the 'artifact_path' matches during logging. Error: {e}")