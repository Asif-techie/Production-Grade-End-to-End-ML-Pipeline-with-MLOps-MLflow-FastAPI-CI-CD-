import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402
from src.data import load_heart_data  # noqa: E402


def test_load_heart_data_shape():
    X, y, _ = load_heart_data(run_eda=False)
    assert X.shape[0] == y.shape[0]


def test_no_missing_values():
    X, y, _ = load_heart_data(run_eda=False)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0
