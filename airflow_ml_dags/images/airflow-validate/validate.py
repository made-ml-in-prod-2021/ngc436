import os
import click
from joblib import load
import logging
import yaml

DEFAUL_LOGGING_CONFIG_FILEPATH = "logging.conf.yaml"

# TODO: set logger
def setup_logging():
    """Reading logger config from yaml"""
    with open(DEFAUL_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))

@click.command("validate")
@click.option("--input-dir")
@click.option("--output-dir")
def validate(input_dir: str, output_model_dir: str):
    pass