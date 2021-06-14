import os

import click
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input-dir")
@click.option("--output-dir")
def predict(input_dir: str, output_dir: str):
    mlflow.set_tracking_uri("http://192.168.1.58:5000")
    MODEL_NAME = os.environ["MODEL_NAME"]

    data_X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    data_y = pd.read_csv(os.path.join(input_dir, "target.csv"))

    client = MlflowClient()
    reg_model = client.get_registered_model(MODEL_NAME)
    try:
        latest_version = [v for v in reg_model.latest_versions if v.current_stage == "Production"][-1]
    except IndexError:
        raise

    local_path = client.get_model_version_download_uri(MODEL_NAME, latest_version.version)
    model = mlflow.sklearn.load_model(local_path)

    predictions = model.predict(data_X, data_y)
    data_y['res'] = predictions
    data_y[['res']].to_csv(os.path.join(output_dir, 'predictions.csv'))


if __name__ == "__main__":
    cli()
