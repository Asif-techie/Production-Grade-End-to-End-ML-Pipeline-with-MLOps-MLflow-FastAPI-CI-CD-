# tests/test_pipeline.py
import pytest

from src.pipeline import pipelines


def test_pipeline_keys():
    assert "LogisticRegression" in pipelines
    assert "RandomForest" in pipelines
