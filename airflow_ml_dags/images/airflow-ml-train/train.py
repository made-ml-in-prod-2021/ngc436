import os

import click
import mlflow
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input-dir")
@click.option("--output-model-dir")
def train(input_dir: str, output_model_dir: str):
    # Not a good solution
    mlflow.set_tracking_uri("http://<set-your-ip>")
    EXP_NAME = os.environ["EXP_NAME"]

    train_X = pd.read_csv(os.path.join(input_dir, "train_X.csv"))
    train_y = pd.read_csv(os.path.join(input_dir, "train_y.csv"))

    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if not exp:
        exp_id = mlflow.create_experiment(EXP_NAME)
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(run_name="model_training", experiment_id=exp_id) as run:
        params = {"n_estimators": 10, "random_state": 42}

        model = RandomForestClassifier(**params)
        mlflow.log_params(params)
        model.fit(train_X, train_y)
        os.makedirs(output_model_dir, exist_ok=True)
        dump(model, os.path.join(output_model_dir, 'model.joblib'))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest"
        )


if __name__ == "__main__":
    cli()
