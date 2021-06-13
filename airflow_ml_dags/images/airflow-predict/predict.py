import os
import click
import pandas as pd
from joblib import load
import logging


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir: str, output_dir: str, model_dir: str):
    data_X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    data_y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    model = load("model.joblib")
    predictions = model.predict(data_X, data_y)
    return predictions
