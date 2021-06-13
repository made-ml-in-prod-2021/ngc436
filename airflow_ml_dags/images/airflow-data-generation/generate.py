import os

import click
from sklearn.datasets import load_breast_cancer
from sdv.tabular import GaussianCopula


@click.command("generate")
@click.argument("output_dir")
def generate_data(output_dir: str, n: int = 500):
    df_dict = load_breast_cancer(as_frame=True)
    data = df_dict['data']
    data['target'] = df_dict['target']
    model = GaussianCopula()
    model.fit(data)
    new_data = model.sample(n)
    os.makedirs(output_dir, exist_ok=True)
    new_data.to_csv(output_dir)
