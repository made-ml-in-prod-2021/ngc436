import logging
import logging.config
import pandas as pd
import yaml

APPLICATION_NAME = "dataset_loading"
DEFAUL_LOGGING_CONFIG_FILEPATH = "logging.conf.yml"
UCI_HEART_DATASET_PATH = "https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
UCI_ATTR_COLUMNS = ['']

# TODO: set default path

def read_csv_data(input_fpath: str, output_fpath: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


