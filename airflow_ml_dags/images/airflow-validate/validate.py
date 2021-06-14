import os
import click
import pandas as pd
from joblib import load
import logging
import yaml

import mlflow
from mlflow.tracking import MlflowClient

from airflow.models import Variable

EXP_NAME = Variable.get("EXP_NAME")
MODEL_NAME = Variable.get("MODEL_NAME")
DEFAUL_LOGGING_CONFIG_FILEPATH = "logging.conf.yaml"

mlflow.set_tracking_uri("http://localhost:5000")


def setup_logging():
    """Reading logger config from yaml"""
    with open(DEFAUL_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir: str):
    setup_logging()
    logger_metrics = logging.getLogger("validation_metrics")

    val_X = pd.read_csv(os.path.join(input_dir, "val_X.csv"))
    val_y = pd.read_csv(os.path.join(input_dir, "val_y.csv"))
    # model = load("model.joblib")

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXP_NAME)
    runs = client.last_run_infos(exp.experiment_id)
    last_run = max(runs, key=lambda x: x.end_time)
    logged_artifacts = client.last_artifacts(last_run.run_id)
    logged_model = logged_artifacts[0]

    mlflow.register_model(f"runs/{last_run.run_id}/{logged_model.path}", MODEL_NAME)

    model = client.get_registered_model(MODEL_NAME)
    latest_version = model.latest_versions[0]

    local_path = client.get_model_version_download_uri(MODEL_NAME, latest_version.version)
    model = mlflow.sklearn.load_model(local_path)
    score = model.score(val_X, val_y)
    logger_metrics.info(f"got metric {score} for model {model_dir.split(' / ')[-1]}")

    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if not exp:
        exp_id = mlflow.create_experiment(EXP_NAME)
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(run_name="model_training", experiment_id=exp_id) as run:
        mlflow.log_metric("mean_accuracy", score)
