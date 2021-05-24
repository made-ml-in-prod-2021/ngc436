from classification.data import read_csv_data, data_train_test_split

TEST_DATASET_PATH = "../../data/raw/heart.csv"
TEST_SIZE = 0.1


def test_read_csv_dataset(dataset_fpath: str = TEST_DATASET_PATH):
    data = read_csv_data(dataset_fpath)
    assert data.shape[0] == 303


def test_train_test_split():
    data = test_read_csv_dataset(TEST_DATASET_PATH)
    train, test = data_train_test_split(data, TEST_SIZE)
    assert test.shape[0] == 30
