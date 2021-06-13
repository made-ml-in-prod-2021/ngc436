import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

import click


@click.command("train")
@click.option("--input-dir")
@click.option("--output-model-dir")
def train(input_dir: str, output_model_dir: str):
    train_X = pd.read_csv(os.path.join(input_dir, "train_X.csv"))
    train_y = pd.read_csv(os.path.join(input_dir, "train_y.csv"))
    model = RandomForestClassifier()
    model.fit(train_X)
    score = model.score(train_X, train_y)
    # TODO: save to log
    dump(model, os.path.join(output_model_dir, 'model.joblib'))
