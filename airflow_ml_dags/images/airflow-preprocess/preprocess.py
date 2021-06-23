import os

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    X_data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y_data = pd.read_csv(os.path.join(input_dir, "target.csv"))

    scaler = StandardScaler()
    scaler.fit(X_data)

    X_train_scaled = scaler.transform(X_data)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_data.columns)

    os.makedirs(output_dir, exist_ok=True)

    X_train_scaled_df.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y_data.to_csv(os.path.join(output_dir, "target.csv"))


if __name__ == "__main__":
    cli()
