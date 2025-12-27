# tests/test_data.py

import sys
import os

# Add src folder to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data import load_heart_data  # now import directly

def test_load_heart_data_shape():
    X, y, _ = load_heart_data(run_eda=False)
    assert X.shape[0] == y.shape[0]

def test_no_missing_values():
    X, y, _ = load_heart_data(run_eda=False)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0
