import numpy as np
import pandas as pd
import pytest

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def test_predict_pipeline_output():
    """
    Test predict pipeline with dummy input data.
    """

    # Create dummy input matching Boston Housing feature set
    custom_input = CustomData(
        crim=0.1, zn=18.0, indus=2.3, chas=0, nox=0.5, rm=6.5,
        age=30.0, dis=5.0, rad=1, tax=300.0, ptratio=15.0, b=396.9, lstat=4.5
    )

    df = custom_input.get_data_as_dataframe()

    pipeline = PredictPipeline()

    try:
        result = pipeline.predict(df)
    except Exception as e:
        pytest.fail(f"Prediction failed: {e}")

    assert isinstance(result, np.ndarray), "Prediction output is not a numpy array"
    assert result.shape == (1,), f"Expected 1 prediction, got shape: {result.shape}"
