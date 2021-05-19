import click
from dataclasses import dataclass
from data import read_csv_data, data_train_test_split
from features import features_processing
from marshmallow_dataclass import class_schema
import yaml
from models import train_sklearn_model

DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass()
class SplittingParams:
    val_size: float
    random_state: int


@dataclass()
class TrainParams:
    model_type: str
    target_col: str
    random_state: int


@dataclass()
class ExperimentParams:
    input_fpath: str
    output_model_fpath: str
    splitting_params: SplittingParams
    train_params: TrainParams


ExperimentPipelineParamsSchema = class_schema(ExperimentParams)


def read_config_file(path: str) -> ExperimentParams:
    with open(path, "r") as file:
        schema = ExperimentPipelineParamsSchema()
        return schema.load(yaml.safe_load(file))


@click.command()
@click.option('--config-path', default=DEFAULT_CONFIG_PATH, help='Set the path to experiments config file')
def run_full_pipeline(config_path):
    params = read_config_file(config_path)
    data = read_csv_data(params.input_fpath)
    train, test = data_train_test_split(data, params.splitting_params.val_size, params.splitting_params.random_state)
    train, test = features_processing(train, test, params.train_params.target_col)
    train_sklearn_model(train, test, params.train_params.target_col, params.train_params.model_type,
                        params.output_model_fpath, params.train_params.random_state)


if __name__ == "__main__":
    run_full_pipeline()

