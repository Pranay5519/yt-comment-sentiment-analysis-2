import mlflow
import pytest
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
load_dotenv() 
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# Set your remote tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow")

@pytest.mark.parametrize("model_name, alias", [
    ("ligbm_model_v1", "production"),
])
def test_load_latest_staging_model(model_name, alias):
    client = MlflowClient()
    
    try:
        # 1. Verify the alias exists via Client
        version_details = client.get_model_version_by_alias(model_name, alias)
        assert version_details is not None
        
        # 2. Use the correct MLflow 3 Alias URI format: models:/name@alias
        # Note: No forward slash after the model name when using @
        model_uri = f"models:/{model_name}@{alias}"
        
        # 3. Use lightgbm.load_model or pyfunc.load_model
        model = mlflow.lightgbm.load_model(model_uri)

        # Assertions
        assert model is not None, "Model object is None"
        print(f"Model '{model_name}' version {version_details.version} loaded successfully via @{alias}.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")