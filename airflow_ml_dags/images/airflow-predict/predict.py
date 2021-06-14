import os
import click
import pandas as pd
from joblib import load
from airflow.models import Variable

from mlflow.tracking import MlflowClient

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
EXP_NAME = Variable.get("EXP_NAME")
MODEL_NAME = Variable.get("MODEL_NAME")


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data_X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    data_y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    # model = load(os.path.join(model_dir, "model.joblib"))

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
