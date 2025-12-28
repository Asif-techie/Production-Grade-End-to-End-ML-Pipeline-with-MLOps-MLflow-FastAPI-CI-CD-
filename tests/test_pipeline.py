import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import pipelines


def test_pipeline_exists():
    assert "LogisticRegression" in pipelines
    assert "RandomForest" in pipelines


def test_pipeline_structure():
    pipe = pipelines["LogisticRegression"]
    step_names = [name for name, _ in pipe.steps]
    assert "select" in step_names
    assert "scaler" in step_names
    assert "smote" in step_names
    assert "clf" in step_names