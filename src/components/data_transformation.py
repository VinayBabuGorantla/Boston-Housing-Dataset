import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Returns a preprocessing pipeline that handles:
        - Missing value imputation with median
        - Standard scaling of numeric columns
        """
        try:
            # Define numerical columns for the Boston dataset
            numerical_features = [
                'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'
            ]

            # Define preprocessing pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_features)
            ])

            logging.info("Preprocessing pipeline created for numerical features.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test CSVs loaded successfully.")

            target_column = "medv"  # Target variable in Boston dataset

            input_features = train_df.columns.drop(target_column).tolist()

            preprocessor = self.get_data_transformer_object()

            # Separate features and target
            X_train = train_df[input_features]
            y_train = train_df[target_column]
            X_test = test_df[input_features]
            y_test = test_df[target_column]

            # Apply transformations
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Preprocessing applied on training and test sets.")

            # Combine features and targets
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save preprocessor object
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logging.info(f"Preprocessor object saved at {self.config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
