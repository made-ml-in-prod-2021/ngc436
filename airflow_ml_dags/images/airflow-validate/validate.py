import os
import click
import pandas as pd
from joblib import load
import logging
import yaml

import mlflow
from mlflow.tracking import MlflowClient

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
    model = load("model.joblib")

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXP_NAME)

    model_ds = model_dir.split('/')[-1]
    score = model.score(val_X, val_y)

    logger_metrics.info(f'got metric {score} for model {model_ds}')
