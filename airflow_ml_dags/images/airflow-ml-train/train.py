import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import click

def _x_y_util(data):
    y = data['target']
    x = data.drop('target')
    return x, y

@click.command("train")
@click.option("--input-dir")
@click.option("--output-model-dir")
def train(input_dir: str, output_model_dir: str):
    train_data = pd.read_csv(os.path.join(input_dir, f"data_train_{{ ds }}.csv"))
    X_train, y_train = _x_y_util()
    test_data = pd.read_csv(os.path.join(input_dir, f"data_test_{{ ds }}.csv"))
    model = RandomForestClassifier()
    model.fit(train_data)
    model.score()
    # TODO
