# tests/test_pipeline.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline import pipelines  # direct import

def test_pipeline_keys():
    expected_keys = ["LogisticRegression", "RandomForest"]
    for key in expected_keys:
        assert key in pipelines
