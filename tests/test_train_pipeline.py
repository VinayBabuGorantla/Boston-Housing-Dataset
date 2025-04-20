import os
import pytest

from src.pipeline import train_pipeline
from src.exception import CustomException

def test_train_pipeline_runs():
    """
    Ensure the training pipeline runs without crashing.
    """
    try:
        # Simulate running the training pipeline
        exec(open("src/pipeline/train_pipeline.py").read())
    except CustomException as ce:
        pytest.fail(f"Training pipeline failed with CustomException: {ce}")
    except Exception as e:
        pytest.fail(f"Training pipeline failed with unexpected error: {e}")

    # Check if trained model was saved
    assert os.path.exists("artifacts/model.pkl"), "Trained model file not found!"
