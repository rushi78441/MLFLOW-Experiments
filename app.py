# Wine quality prediction with MLflow & DAGsHub
import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

#  Initialize DAGsHub integration (this sets tracking URI and credentials automatically) - for tracking model in remote server
import dagshub
dagshub.init(repo_owner='rushi78441', repo_name='MLFLOW-Experiments', mlflow=True)

## set logger configuration
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

## model performance evaluations
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV. Error: %s", e)
        sys.exit(1)

    # Split into train/test sets
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Read alpha/l1_ratio from command line or use default sys.argv is positional argument , this refers to cmd command python app.py position1_value(alpha) position2_value(l1)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


    ## Start MLflow tracking
    mlflow.set_experiment("wine-quality-prediction")
    with mlflow.start_run():
        # Train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        # Predict
        predicted_qualities = model.predict(test_x)

        # Evaluate metrics
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        signature = infer_signature(train_x, predicted_qualities)
        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "model",
            signature = signature,
            await_registration_for=0,  # ⛔ Prevent MLflow 3.x registry feature
            input_example = train_x.iloc[:1]
        )