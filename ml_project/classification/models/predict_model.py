import pandas as pd
from joblib import load


def predict_model(df: pd.DataFrame, model_name: str):
    cls = load(model_name)
    return cls.predict(df)
