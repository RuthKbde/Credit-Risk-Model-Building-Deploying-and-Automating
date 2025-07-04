Credit Risk Model Building, Deployment, and Automation
This repository contains the end-to-end implementation of a credit risk classification model, from data preprocessing and feature engineering to model training, deployment as an API, and setting up a Continuous Integration/Continuous Deployment (CI/CD) pipeline.

Project Overview
In the financial sector, accurate credit risk assessment is crucial for regulatory compliance (e.g., Basel II Accord) and sound business decisions. This project addresses the challenge of building a credit risk model when explicit default labels are unavailable, by engineering a proxy target variable. It demonstrates a robust MLOps workflow, ensuring the model is not only performant but also maintainable and deployable.

Key Features
Data Processing & Feature Engineering: Automated scripts for cleaning raw transactional data and creating relevant features (time-based, aggregate customer metrics).

Proxy Target Variable Creation: Utilizes RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to identify "high-risk" (disengaged) customers as a proxy for default.

Model Training & Tracking:

Comparison of Logistic Regression and Random Forest Classifiers.

Hyperparameter tuning using GridSearchCV.

Comprehensive model evaluation using metrics suitable for imbalanced datasets (Precision, Recall, F1-Score, ROC-AUC).

Experiment tracking and model registration via MLflow.

Unit Testing: Robust unit tests for data processing components using pytest.

Containerized API Deployment:

RESTful API built with FastAPI to serve real-time credit risk predictions.

Data validation using Pydantic models.

Model loading directly from MLflow Model Registry.

Dockerization: Dockerfile and docker-compose.yml for containerizing the FastAPI service, ensuring portability and consistent environments.

CI/CD Pipeline: GitHub Actions workflow for automated code linting (flake8) and unit testing on every push to the main branch, enforcing code quality.

Business Understanding & Model Trade-offs
The project addresses the need for interpretable and well-documented models, as emphasized by the Basel II Accord. The absence of a direct "default" label necessitated the creation of a proxy variable (is_high_risk) based on customer disengagement. While enabling model development, this proxy introduces business risks, as it may not perfectly capture true default behavior, requiring continuous monitoring.

The choice between simple (e.g., Logistic Regression) and complex (e.g., Random Forest) models involves a trade-off between interpretability and performance. Logistic Regression offers transparency for regulatory compliance, while Random Forest provides higher predictive power but can be less interpretable. This project explores both to find a balance suitable for a financial context.

Project Structure
.
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD workflow
├── data/
│   ├── data.csv               # Raw input data
│   └── processed/
│       ├── processed_transactions.csv # Processed features
│       └── processed_data_with_target.csv # Processed data with proxy target
├── mlruns/                    # MLflow tracking server data
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory Data Analysis notebook
├── reports/
│   └── figures/               # Saved EDA visualizations
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI application
│   │   └── pydantic_models.py # Pydantic data models for API
│   ├── data_processing.py     # FeatureExtractor and CustomerAggregator classes
│   ├── create_target_variable.py # Script for RFM and clustering
│   └── train.py               # Model training and MLflow tracking script
├── tests/
│   ├── __init__.py
│   ├── simple_test.py         # Basic pytest functionality test
│   └── test_data_processing.py # Unit tests for data processing
├── Dockerfile                 # Docker build instructions for the API
├── docker-compose.yml         # Docker Compose configuration
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file

Setup and Installation
Prerequisites
Python 3.9+

Conda (recommended for environment management)

Docker Desktop (for containerization)

Git

1. Clone the Repository
git clone https://github.com/RuthKbde/Credit-Risk-Model-Building-Deploying-and-Automating.git
cd Credit-Risk-Model-Building-Deploying-and-Automating

2. Create and Activate Conda Environment
conda create --name credit_risk_env python=3.9
conda activate credit_risk_env

3. Install Dependencies
pip install -r requirements.txt

(Note: If you encounter issues with pip install -r requirements.txt due to conda-specific builds, refer to the project's documentation or contact the maintainer for a cleaned requirements.txt.)

4. Prepare Data
Place your raw data.csv file into the data/ directory.

Usage
1. Run Data Processing and Feature Engineering
python src/data_processing.py

2. Create Proxy Target Variable
python src/create_target_variable.py

3. Train and Track Model
python src/train.py

This will train the models, log experiments to mlruns/, and register the best model in the MLflow Model Registry.

4. Run MLflow UI (Optional)
To view experiment runs and registered models:

mlflow ui

Then open http://localhost:5000 in your web browser.

5. Build and Run the API Service (with Docker)
Ensure Docker Desktop is running.

docker compose build
docker compose up

The API will be accessible at http://localhost:8000. You can view the interactive documentation at http://localhost:8000/docs.

6. Run Unit Tests
pytest

CI/CD Pipeline
The project uses GitHub Actions for Continuous Integration. The workflow defined in .github/workflows/ci.yml triggers on every push to the main branch and performs:

Code Linting (flake8): Checks for code style and potential errors.

Unit Tests (pytest): Executes all project unit tests.

The build will fail if either the linter or tests fail, ensuring code quality and preventing regressions.
