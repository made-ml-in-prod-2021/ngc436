import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import NoReturn
from joblib import dump


def train_sklearn_model(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        target: str,
                        model_type: str,
                        model_fpath: str,
                        random_state: int) -> NoReturn:
    if model_type == "RandomForestClassifier":
        cls = RandomForestClassifier(random_state=random_state)
    elif model_type == "DecisionTreeClassifier":
        cls = DecisionTreeClassifier(random_state=random_state)
    else:
        print("There is no support for such model")
        return 1
    train_cols = list(train_df)
    train_cols.remove(target)
    cls.fit(train_df[train_cols], train_df[target])
    score = cls.score(test_df[train_cols], test_df[target])
    dump(cls, model_fpath)
    return score
