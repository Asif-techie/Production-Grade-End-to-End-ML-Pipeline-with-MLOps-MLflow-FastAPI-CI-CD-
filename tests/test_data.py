# tests/test_data.py

from src.data import load_heart_data


def test_load_heart_data_shape():
    X, y = load_heart_data()
    assert X.shape[0] == y.shape[0]


def test_no_missing_values():
    X, y = load_heart_data()
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0
