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

import dagshub

# 🔧 Initialize DAGsHub integration (this sets tracking URI and credentials automatically) - for tracking model in remote server
dagshub.init(repo_owner='rushi78441', repo_name='MLFLOW-Experiments', mlflow=True)

# Logging setup
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 📥 Load dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV. Error: %s", e)
        sys.exit(1)

    # 🔀 Split into train/test sets
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # ⚙️ Read alpha/l1_ratio from command line or use default sys.argv is positional argument , this refers to cmd command python app.py position1_value(alpha) position2_value(l1)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # 🚀 Start MLflow run
    with mlflow.start_run():
        # Train model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # 📊 Print metrics
        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # 📌 Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ✅ Infer signature and log model
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            signature=signature,
            input_example=train_x.iloc[:1]
        )

