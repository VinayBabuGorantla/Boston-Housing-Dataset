# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import mlflow
import mlflow.sklearn

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: any, test_array: any) -> float:
        try:
            logging.info("Starting model training component.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models: Dict[str, any] = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
            }

            params: Dict[str, dict] = {
                "Linear Regression": {},  # No hyperparams for GridSearch
                "Decision Tree": {
                    'criterion': ['squared_error'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R² score: {best_score}")

            if best_score < 0.7:
                raise CustomException(f"No suitable model found. Best R²: {best_score}", sys)

            # MLflow experiment logging
            mlflow.set_experiment("Boston_Housing_Regression")

            with mlflow.start_run():
                mlflow.log_param("model_name", best_model_name)
                for param_name, param_value in params[best_model_name].items():
                    mlflow.log_param(param_name, param_value)

                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                final_r2 = r2_score(y_test, y_pred)

                mlflow.log_metric("r2_score", final_r2)
                mlflow.sklearn.log_model(best_model, "model")

                logging.info(f"MLflow logging complete. Final R²: {final_r2:.4f}")

            save_object(self.config.trained_model_path, best_model)
            logging.info(f"Trained model saved at: {self.config.trained_model_path}")

            return final_r2

        except Exception as e:
            raise CustomException(e, sys)
