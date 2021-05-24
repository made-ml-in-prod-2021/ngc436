from classification.data import read_csv_data, data_train_test_split
from classification.features import features_processing
from classification.models import train_sklearn_model

TEST_DATASET_PATH = "../../data/raw/heart.csv"
TEST_SIZE = 0.2
RANDOM_STATE_SPLIT = 3
RANDOM_STATE = 4
TARGET = "target"
MNAME = "RandomForestClassifier"
OUT = "models/model_rf.pkl"

def test_full_pipeline():
    data = read_csv_data(TEST_DATASET_PATH)
    train, test = data_train_test_split(data, TEST_SIZE, RANDOM_STATE_SPLIT)
    train, test = features_processing(train, test, TARGET)
    score = train_sklearn_model(train, test, TARGET, MNAME,
                        OUT, RANDOM_STATE)

    assert score > 0.8
