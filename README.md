# 🏠 Boston Housing Price Prediction - End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black)

## 📊 Overview

This is a production-ready **Machine Learning Regression project** that predicts housing prices in Boston. It demonstrates a complete **MLOps workflow**, integrating data versioning, experiment tracking, CI/CD pipelines, and containerized deployment.

The application features a **modern, responsive UI** for real-time predictions.

## 🚀 Key Features

- **End-to-End Pipeline**: Automated data ingestion, transformation, and model training.
- **Data Version Control (DVC)**: Reproducible pipelines with `dvc.yaml`.
- **Experiment Tracking**: MLflow integration to track model performance and parameters.
- **CI/CD**: GitHub Actions pipeline for automated testing, linting, and Docker build/push.
- **Containerization**: Fully Dockerized application for consistent deployment.
- **Modern UI**: Polished, responsive web interface built with Bootstrap 5 and custom CSS.
- **Code Quality**: Linting with `flake8` and unit testing with `pytest`.

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Web Framework**: Flask
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **MLOps Tools**: DVC, MLflow, Docker, GitHub Actions
- **Frontend**: HTML5, CSS3, Bootstrap 5

## ⚙️ Installation & Local Run

1.  **Clone the repository**
    ```bash
    git clone https://github.com/VinayBabuGorantla/Boston-Housing-Dataset.git
    cd Boston-Housing-Dataset
    ```

2.  **Create a virtual environment**
    ```bash
    conda create -p venv python=3.11 -y
    conda activate ./venv
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the DVC Pipeline** (Optional, to retrain)
    ```bash
    dvc repro
    ```

5.  **Start the Application**
    ```bash
    python app.py
    ```
    Open [http://localhost:8080](http://localhost:8080) in your browser.

## 🐳 Running with Docker

1.  **Build the image**
    ```bash
    docker build -t boston-housing-app .
    ```

2.  **Run the container**
    ```bash
    docker run -p 8080:8080 boston-housing-app
    ```

## 🔄 DVC Pipeline Stages

The pipeline is defined in `dvc.yaml` and consists of three stages:

1.  **`data_ingestion`**: Loads data from source, splits into train/test, and saves to `artifacts/`.
2.  **`data_transformation`**: Cleans data, handles missing values, scales features, and saves the preprocessor.
3.  **`model_trainer`**: Trains multiple models (Linear Regression, Decision Tree, Random Forest), logs best model to MLflow, and saves it as `model.pkl`.

## 🧪 Testing

Run unit tests to verify the pipeline components:
```bash
pytest
```

## 🤝 Contributing

1.  Fork the repo
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
