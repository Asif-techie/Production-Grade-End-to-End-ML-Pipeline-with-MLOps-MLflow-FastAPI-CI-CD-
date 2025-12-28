import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402
from src.pipeline import pipelines  # noqa: E402


def test_pipeline_keys():
    assert "LogisticRegression" in pipelines
    assert "RandomForest" in pipelines
