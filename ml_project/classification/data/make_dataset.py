import logging
import logging.config
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from typing import Tuple

APPLICATION_NAME = "dataset_loading"
DEFAUL_LOGGING_CONFIG_FILEPATH = "../logging.conf.yml"
UCI_ATTR_COLUMNS = ['']

logger = logging.getLogger(APPLICATION_NAME)  # singleton


def setup_logging():
    with open(DEFAUL_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_data(input_fpath: str) -> pd.DataFrame:
    # logger = logging.getLogger(__name__)
    try:
        data = pd.read_csv(input_fpath)
    except FileNotFoundError:
        print(f"Path {input_fpath} is not valid")
        return 1
    return data


def data_train_test_split(df: pd.DataFrame, val_size: float, random_state: int = 11) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=val_size, random_state=random_state)
    return train, test
