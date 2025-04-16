import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info(">>> Training pipeline started.")

        # Step 1: Ingest data
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Transform data
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

        # Step 3: Train model
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"✅ Training pipeline completed successfully with R² score: {r2_score:.4f}")

    except Exception as e:
        raise CustomException(e, sys)
