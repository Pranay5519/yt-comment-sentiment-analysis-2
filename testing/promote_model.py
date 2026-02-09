from mlflow.tracking import MlflowClient
import mlflow
import os
from dotenv import load_dotenv
load_dotenv() 
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


def promote_staging_to_production(model_name: str):
    mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow")


    client = MlflowClient(tracking_uri= "https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow")

    try:
        # Get current production and archive it
        try:
            old_production = client.get_model_version_by_alias(model_name, "production")
            client.set_registered_model_alias(
                model_name, "archived", old_production.version
            )
            print(f"Archived old production version: {old_production.version}")
        except Exception:
            print("No existing production model found.")

        # Get staging version and promote it
        try:
            staging_version = client.get_model_version_by_alias(
                model_name, "staging"
            ).version

            # Promote to production
            client.set_registered_model_alias(
                model_name, "production", staging_version
            )

            # Remove staging alias
            client.delete_registered_model_alias(model_name, "staging")

            print(f"Promoted version {staging_version} to production")

        except Exception as e:
            print(f"No staging model found: {e}")
            raise

    except Exception as e:
        print(f"Promotion failed: {e}")
        raise


# Usage
promote_staging_to_production("light_gbm_model")
