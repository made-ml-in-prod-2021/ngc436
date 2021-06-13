import os

import click
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sdv.tabular import GaussianCopula

def write_to_file(output_dir: str, data: pd.DataFrame):
    full_path = os.path.join(output_dir, "{{ ds }}")
    os.makedirs(full_path, exist_ok=True)
    y = data[['target']]
    y.to_csv(os.path.join(full_path, "traget.csv"), index=None)
    data.drop("target")
    data.to_csv(os.path.join(full_path, "data.csv"), index=None)


@click.command("generate")
@click.argument("output_dir")
def generate_data(output_dir: str, n: int = 500):
    df_dict = load_breast_cancer(as_frame=True)
    data = df_dict['data']
    data['target'] = df_dict['target']
    model = GaussianCopula()
    model.fit(data)
    new_data = model.sample(n)
    write_to_file(output_dir, new_data)
