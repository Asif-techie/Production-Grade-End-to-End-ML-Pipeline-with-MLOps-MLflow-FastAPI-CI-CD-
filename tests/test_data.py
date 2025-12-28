import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_heart_data


def test_load_heart_data_shapes():
    X, y, df = load_heart_data(run_eda=False)
    assert X.shape[0] == y.shape[0]
    assert df.shape[0] > 0


def test_no_missing_values_after_cleaning():
    X, y, _ = load_heart_data(run_eda=False)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0