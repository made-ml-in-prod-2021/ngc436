import pandas as pd
import numpy as np
from sklearn import preprocessing
from typing import Tuple


def features_processing(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    float_cols = list(df_train.select_dtypes(include=[np.float]).columns.values)
    float_cols = list(set(float_cols).difference(set(target)))
    min_max_scaler = preprocessing.MinMaxScaler()
    df_train[float_cols] = min_max_scaler.fit_transform(df_train[float_cols])
    df_test[float_cols] = min_max_scaler.transform(df_test[float_cols])
    return df_train, df_test
