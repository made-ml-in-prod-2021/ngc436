import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("train-test")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--val-size")
def train_test_preparation(input_dir: str,
                           output_dir: str, val_size: float):
    data_X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    data_y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    X_train, X_val, y_train, y_val = train_test_split(
        data_X, data_y, test_size=val_size, stratify=data_y)

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "train_X.csv"), index=None)
    X_val.to_csv(os.path.join(output_dir, "val_X.csv"), index=None)
    y_train.to_csv(os.path.join(output_dir, "train_y.csv"), index=None)
    y_val.to_csv(os.path.join(output_dir, "val_y.csv"), index=None)
