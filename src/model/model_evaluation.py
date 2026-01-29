import numpy as np
import pandas as pd
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
import dagshub
from dotenv import load_dotenv
import logging   # 🔴 added

# -------------------- LOGGING SETUP --------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("model_evaluation.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
# ------------------------------------------------------

# Load environment variables
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    logger.error("DAGSHUB_PAT environment variable is not set")
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/Pranay5519/yt-comment-sentiment-analysis-2.mlflow"
)


def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug(f"Data shape: {df.shape}")
    return df


def load_model(model_path: str):
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    logger.info(f"Loading vectorizer from {vectorizer_path}")
    with open(vectorizer_path, 'rb') as file:
        return pickle.load(file)


def load_params(params_path: str) -> dict:
    logger.info(f"Loading params from {params_path}")
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray):
    logger.info("Evaluating model")
    logger.debug(f"X_test shape: {X_test.shape}")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm


def log_confusion_matrix(cm, dataset_name):
    logger.info("Logging confusion matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(local_path=cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    logger.info("Saving model info")
    model_info = {
        "run_id": run_id,
        "model_path": model_path
    }
    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)


# -------------------- LOAD STATIC DATA --------------------
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

params = load_params(
    os.path.join(root_dir, "params.yaml")
)

test_data = load_data(
    os.path.join(root_dir, "data/interim/test_processed.csv")
)

dataset = mlflow.data.from_pandas(test_data, name="evaluation_set")
# --------------------------------------------------------


def main():
    logger.info("Starting MLflow evaluation run")
    mlflow.set_experiment("dvc-pipeline-1")

    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        for key, value in params.items():
            mlflow.log_param(key, value)

        model = load_model(
            os.path.join(root_dir, "lgbm_model.pkl")
        )
        vectorizer = load_vectorizer(
            os.path.join(root_dir, "tfidf_vectorizer.pkl")
        )

        X_test_tfidf = vectorizer.transform(test_data["clean_comment"].values)

        X_test_df = pd.DataFrame(
            X_test_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        y_test = test_data["category"].values

        input_example = X_test_df[:5].astype("float32")

        signature = infer_signature(
            input_example,
            model.predict(input_example)
        )

        mlflow.lightgbm.log_model(
            model,
            name="lgbm_model",
            signature=signature,
            input_example=input_example,
            registered_model_name="ligbm_model_v1"
        )

        save_model_info(run.info.run_id, "lgbm_model", "experiment_info.json")

        mlflow.log_artifact(
            local_path=os.path.join(root_dir, "tfidf_vectorizer.pkl")
        )

        mlflow.log_input(dataset, context="evaluation")

        report, cm = evaluate_model(model, X_test_df, y_test)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metrics({
                    f"test_{label}_precision": metrics["precision"],
                    f"test_{label}_recall": metrics["recall"],
                    f"test_{label}_f1-score": metrics["f1-score"]
                })

        log_confusion_matrix(cm, "Test Data")

        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("task", "Sentiment Analysis")
        mlflow.set_tag("dataset", "YouTube Comments")

        logger.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()
