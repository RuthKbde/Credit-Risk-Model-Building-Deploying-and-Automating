import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

print("Script started: Importing libraries and setting up paths.")

# Configure logging for better visibility during MLflow runs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration and Constants ---
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_data_with_target.csv')

# MLflow setup
# Set MLflow tracking URI to a local directory (mlruns folder will be created in project root)
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Credit_Risk_Model_Training")

# --- Main Training Function ---

def train_and_track_models():
    """
    Loads processed data, splits it, trains multiple models with hyperparameter tuning,
    evaluates them, and tracks results using MLflow. Registers the best model.
    """
    logging.info("Starting model training and tracking process...")

    # 1. Load Processed Data
    try:
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        logging.info(f"Processed data loaded successfully from: {PROCESSED_DATA_PATH}")
        logging.info(f"Processed data shape: {df_processed.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}.")
        logging.error("Please ensure Task 4 has been completed and 'processed_data_with_target.csv' is saved.")
        return

    # 2. Separate Features (X) and Target (y)
    # The target variable is 'is_high_risk'
    # All other columns are features, except for original IDs and TransactionStartTime
    
    # Identify feature columns and target column
    target_column = 'is_high_risk'
    
    # Exclude original ID columns and TransactionStartTime (as new time features are extracted)
    # These should be in 'remainder__' prefixed columns
    cols_to_exclude_from_features = [
        'remainder__TransactionId', 'remainder__BatchId', 'remainder__AccountId',
        'remainder__SubscriptionId', 'remainder__CustomerId',
        'remainder__TransactionStartTime', target_column
    ]
    
    # Filter out columns that are not in the DataFrame (e.g., if some IDs were not passed through)
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude_from_features]

    X = df_processed[feature_columns]
    y = df_processed[target_column]
    
    # Convert all object dtypes in X to numeric if possible (from ColumnTransformer output)
    # This is crucial as ColumnTransformer can sometimes output 'object' dtype for numerical columns
    # if there are mixed types or specific configurations.
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='raise') # 'raise' will error if conversion fails
            except ValueError:
                logging.warning(f"Column '{col}' could not be converted to numeric. Keeping as object.")
    
    # Ensure target variable is integer
    y = y.astype(int)

    logging.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    logging.info(f"Target variable distribution:\n{y.value_counts(normalize=True)}")

    # Check for NaN values in X after conversion (if any) and impute if necessary
    # This is a safeguard, as SimpleImputer in the pipeline should have handled original NaNs.
    if X.isnull().sum().sum() > 0:
        logging.warning("NaN values found in features after loading. Imputing with median as a safeguard.")
        for col in X.select_dtypes(include=np.number).columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

    # 3. Split the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class distribution
    )
    logging.info(f"Data split: X_train {X_train.shape}, X_test {X_test.shape}")
    logging.info(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")
    logging.info(f"y_test distribution:\n{y_test.value_counts(normalize=True)}")

    # Models to train and their hyperparameter grids for GridSearchCV
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'), # 'balanced' for imbalance
            "params": {
                'C': [0.01, 0.1, 1, 10]
            }
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'), # 'balanced' for imbalance
            "params": {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        }
    }

    best_model = None
    best_roc_auc = -1
    best_model_name = ""

    # 4. Train Models with Hyperparameter Tuning and Track with MLflow
    for model_name, config in models.items():
        model = config["model"]
        params = config["params"]

        logging.info(f"\n--- Training {model_name} ---")

        # Start an MLflow run for each model
        with mlflow.start_run(run_name=f"{model_name}_GridSearch"):
            # Log model parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("stratify", True)

            # Hyperparameter Tuning using GridSearchCV
            grid_search = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_estimator = grid_search.best_estimator_
            logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logging.info(f"Best ROC-AUC score on CV for {model_name}: {grid_search.best_score_:.4f}")

            # Log best parameters found by Grid Search
            mlflow.log_params({f"best_param_{k}": v for k, v in grid_search.best_params_.items()})
            mlflow.log_metric(f"best_cv_roc_auc", grid_search.best_score_)

            # 5. Model Evaluation on Test Set
            y_pred = best_estimator.predict(X_test)
            y_proba = best_estimator.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            logging.info(f"Test Metrics for {model_name}:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall: {recall:.4f}")
            logging.info(f"  F1-Score: {f1:.4f}")
            logging.info(f"  ROC-AUC: {roc_auc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_roc_auc": roc_auc
            })

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                artifact_path=f"model_{model_name.lower()}",
                # Register the model if it's the best one found so far
                registered_model_name=None # Will register later if it's the absolute best
            )

            # Determine the best model for registration
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = best_estimator
                best_model_name = model_name
                logging.info(f"New best model found: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

    # 6. Model Registration (Best Model)
    if best_model:
        logging.info(f"\nRegistering the best model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
        # Start a new MLflow run specifically for registering the final best model
        with mlflow.start_run(run_name=f"Register_Best_Model_{best_model_name}"):
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"final_best_model_{best_model_name.lower()}",
                registered_model_name="CreditRiskClassifier" # Register under a consistent name
            )
            mlflow.log_metric("final_best_roc_auc", best_roc_auc)
            logging.info(f"Model '{best_model_name}' registered as 'CreditRiskClassifier' in MLflow Model Registry.")
    else:
        logging.warning("No best model identified for registration.")

    logging.info("Model training and tracking process completed.")

if __name__ == "__main__":
    train_and_track_models()