import os
import sys
import pandas as pd

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction pipeline...")

            # Load model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info("Model and preprocessor loaded successfully.")

            # Preprocess input features
            transformed_data = preprocessor.transform(features)
            logging.info(f"Input features transformed. Shape: {transformed_data.shape}")

            # Predict
            preds = model.predict(transformed_data)
            logging.info("Prediction completed.")

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Collects all features for Boston Housing model and prepares a DataFrame.
    """

    def __init__(
        self, crim, zn, indus, chas, nox, rm, age,
        dis, rad, tax, ptratio, b, lstat
    ):
        self.feature_dict = {
            "crim": crim,
            "zn": zn,
            "indus": indus,
            "chas": chas,
            "nox": nox,
            "rm": rm,
            "age": age,
            "dis": dis,
            "rad": rad,
            "tax": tax,
            "ptratio": ptratio,
            "b": b,
            "lstat": lstat
        }

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.DataFrame([self.feature_dict])
            df.columns = df.columns.str.lower()  # Normalize column names
            logging.info("Custom data converted to DataFrame successfully.")
            return df
        except Exception as e:
            raise CustomException(e, sys)
