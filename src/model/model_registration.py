import json
import mlflow
import logging
import os
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise
    
def register_logged_model(model_name : str , model_info: dict):
    """Register the model that was Logged in during Evaluation step"""
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pranay5519"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_PAT")
    
    tracking_uri = "https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow"
    client = MlflowClient(tracking_uri=tracking_uri)
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    try:
        # Check if model exists first to avoid unnecessary 403/Conflict errors
        try:
            client.get_registered_model(model_name)
            logger.info(f"Model {model_name} already exists.")
        except:
            logger.info(f"Creating new registered model: {model_name}")
            client.create_registered_model(model_name)
        
        # Create the version
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=model_info['run_id']
        )
        
        # In MLflow 3, we use Aliases. 
        # This replaces the old "Transition to Staging"
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=mv.version
        )
        
        logger.debug(f'Model {model_name} version {mv.version} is now @staging')
        return mv

    except Exception as e:
        logger.error('Registration failed: %s', e)
        raise
    
    
def main():
    try:
        model_info_path = r'D:\yt-comment-sentiment-analysis2\experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "LightGBM-3.0"
        register_logged_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()