import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import click

from airflow.models import Variable

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
EXP_NAME = Variable.get("EXP_NAME")
MODEL_NAME = Variable.get("MODEL_NAME")


@click.command("train")
@click.option("--input-dir")
@click.option("--output-model-dir")
def train(input_dir: str, output_model_dir: str):
    train_X = pd.read_csv(os.path.join(input_dir, "train_X.csv"))

    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if not exp:
        exp_id = mlflow.create_experiment(EXP_NAME)
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(run_name="model_training", experiment_id=exp_id) as run:
        params = {"n_estimators": 10, "random_state": 42}

        model = RandomForestClassifier(**params)
        mlflow.log_params(params)
        model.fit(train_X)
        dump(model, os.path.join(output_model_dir, 'model.joblib'))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest"
        )