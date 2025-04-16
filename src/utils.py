# src/utils.py

import os
import sys
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    try:
        report = {}
        for model_name, model in models.items():
            print(f"Evaluating model: {model_name}")

            if param.get(model_name):
                grid = GridSearchCV(model, param[model_name], cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)

            y_pred_test = best_model.predict(X_test)
            test_score = r2_score(y_test, y_pred_test)
            report[model_name] = test_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
