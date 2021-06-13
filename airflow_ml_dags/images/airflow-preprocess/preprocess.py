import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

import click


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess_data(input_dir: str, output_dir: str):
    X_data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y_data = pd.read_csv(os.path.join(input_dir, "target.csv"))

    scaler = StandardScaler()
    scaler.fit(X_data)

    X_train_scaled = scaler.transform(X_data)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_data.columns)
    X_train_scaled_df.to_csv(os.path.join(output_dir, "data.csv"), index=None)
    y_data.to_csv(os.path.join(output_dir, "target.csv"))
