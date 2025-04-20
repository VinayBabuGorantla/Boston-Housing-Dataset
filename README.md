# 🏠 Boston Housing Price Prediction - MLOps Project

## 📊 Overview

This is an end-to-end **Machine Learning Regression project** that predicts housing prices in Boston based on features such as crime rate, number of rooms, etc., using the **Boston Housing dataset**.

This project is built for production readiness and follows a complete **MLOps workflow**, including:

- ✅ Data Ingestion
- ✅ Data Transformation
- ✅ Model Training with GridSearchCV
- ✅ Model Evaluation and Persistence
- ✅ Flask Web Application
- ✅ Logging and Exception Handling
- ✅ DVC for data and model versioning
- ✅ MLflow for experiment tracking
- ✅ Docker for containerization and deployment

# Docker Build and Run Commands
docker build -t project-name .
docker run -p 8080:8080 project-name

