import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    source_data_path: str = os.path.join("notebook", "data", "boston.csv")  # NEW: External CSV path

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component.")
        try:
            # Load the Boston housing dataset from CSV
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info("Boston Housing dataset loaded from CSV into DataFrame.")

            # Clean column names: strip spaces and convert to lowercase
            df.columns = df.columns.str.strip().str.lower()
            logging.info("Column names standardized: %s", df.columns.tolist())

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(
                f"Train and test data saved at {self.ingestion_config.train_data_path} and {self.ingestion_config.test_data_path}"
            )

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
