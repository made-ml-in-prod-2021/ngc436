import logging
import os
import time
from pprint import pprint

import click
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

DEFAUL_LOGGING_CONFIG_FILEPATH = "/logging.conf.yaml"


def setup_logging():
    """Reading logger config from yaml"""
    with open(DEFAUL_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input-dir")
def validate(input_dir: str):
    mlflow.set_tracking_uri("http://192.168.1.58:5000")
    EXP_NAME = os.environ["EXP_NAME"]
    MODEL_NAME = os.environ["MODEL_NAME"]

    setup_logging()
    logger_metrics = logging.getLogger("validation_metrics")

    val_X = pd.read_csv(os.path.join(input_dir, "val_X.csv"))
    val_y = pd.read_csv(os.path.join(input_dir, "val_y.csv"))
    # model = load("model.joblib")

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXP_NAME)
    runs = client.list_run_infos(exp.experiment_id)
    pprint(runs)
    last_run = max(runs, key=lambda x: x.end_time)

    time.sleep(10)

    logged_artifacts = client.list_artifacts(last_run.run_id)
    logged_model = logged_artifacts[0]

    mlflow.register_model(f"runs/{last_run.run_id}/{logged_model.path}", MODEL_NAME)

    model = client.get_registered_model(MODEL_NAME)
    latest_version = model.latest_versions[0]

    local_path = client.get_model_version_download_uri(MODEL_NAME, latest_version.version)
    model = mlflow.sklearn.load_model(local_path)
    score = model.score(val_X, val_y)
    logger_metrics.info(f"got metric {score} for model")

    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if not exp:
        exp_id = mlflow.create_experiment(EXP_NAME)
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(run_name="model_training", experiment_id=exp_id) as run:
        mlflow.log_metric("mean_accuracy", score)


if __name__ == "__main__":
    cli()
