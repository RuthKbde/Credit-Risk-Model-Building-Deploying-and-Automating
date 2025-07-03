import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os # For path handling in main function

# --- Custom Transformers ---
# These custom transformers allow us to integrate non-standard sklearn operations
# (like date feature extraction and aggregations) directly into our pipeline.

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract time-based features from 'TransactionStartTime'.
    Extracts hour, day, month, and year from the transaction timestamp.
    """
    def __init__(self, date_column='TransactionStartTime'):
        self.date_column = date_column

    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data during fitting
        return self

    def transform(self, X):
        # Ensure we're working on a copy to avoid modifying the original DataFrame
        X_copy = X.copy()

        # Convert the transaction start time to datetime objects
        # errors='coerce' will turn unparseable dates into NaT (Not a Time)
        X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column], errors='coerce')

        # Extract new features
        X_copy['TransactionHour'] = X_copy[self.date_column].dt.hour
        X_copy['TransactionDay'] = X_copy[self.date_column].dt.day
        X_copy['TransactionMonth'] = X_copy[self.date_column].dt.month
        X_copy['TransactionYear'] = X_copy[self.date_column].dt.year

        # Handle potential NaNs introduced by datetime conversion (e.g., if original date was invalid)
        # For simplicity, we'll impute with a common value or 0, but a more robust strategy
        # might be needed depending on the amount of NaNs.
        for col in ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']:
            X_copy[col] = X_copy[col].fillna(-1).astype(int) # Use -1 or another indicator for missing

        return X_copy

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to create aggregate features per customer.
    Calculates total amount, average amount, transaction count, and standard deviation
    of amounts for each CustomerId.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        # No fitting required for this aggregation logic
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Group by customer and calculate aggregate features
        # Ensure to handle potential NaNs from std() if a customer has only one transaction
        customer_aggregates = X_copy.groupby(self.customer_id_col).agg(
            TotalTransactionAmount=(self.amount_col, 'sum'),
            AverageTransactionAmount=(self.amount_col, 'mean'),
            TransactionCount=(self.customer_id_col, 'count'), # Count of transactions per customer
            StdDevTransactionAmounts=(self.amount_col, 'std')
        ).reset_index()

        # Fill NaN in StdDevTransactionAmounts (occurs for customers with only 1 transaction)
        # A standard deviation of 0 makes sense for a single transaction.
        customer_aggregates['StdDevTransactionAmounts'] = customer_aggregates['StdDevTransactionAmounts'].fillna(0)

        # Merge these new features back to the original DataFrame
        # We use a left merge to keep all original transactions and add the customer-level aggregates
        X_copy = pd.merge(X_copy, customer_aggregates, on=self.customer_id_col, how='left')

        return X_copy

# --- Feature Engineering Pipeline Definition ---

def create_feature_engineering_pipeline(numerical_features, categorical_features_onehot):
    """
    Creates and returns a scikit-learn Pipeline for comprehensive feature engineering.

    Args:
        numerical_features (list): List of column names that are numerical and need scaling/imputation.
                                   This list should include both original and newly created numerical features.
        categorical_features_onehot (list): List of column names that are categorical
                                            and need One-Hot Encoding.

    Returns:
        sklearn.pipeline.Pipeline: A configured scikit-learn pipeline for feature engineering.
    """

    # Define preprocessing steps for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Use median for robustness against outliers
        ('scaler', StandardScaler()) # Standardize numerical features (mean=0, std=1)
    ])

    # Define preprocessing steps for categorical features using One-Hot Encoding
    categorical_transformer_onehot = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with the mode
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Convert categories to binary columns
    ])

    # Combine transformers using ColumnTransformer to apply different transformations
    # to different sets of columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat_onehot', categorical_transformer_onehot, categorical_features_onehot)
            # Add other transformers here if you decide to use Label Encoding for specific columns
            # or other custom transformers.
        ],
        remainder='passthrough' # Keep columns not specified in transformers (e.g., IDs, target variable)
    )

    # Define the full feature engineering pipeline
    # The order of steps is crucial:
    # 1. Feature Extraction (e.g., from dates)
    # 2. Customer Aggregations
    # 3. Preprocessing (Imputation, Encoding, Scaling)
    feature_engineering_pipeline = Pipeline(steps=[
        ('feature_extractor', FeatureExtractor()),
        ('customer_aggregator', CustomerAggregator()),
        ('preprocessor', preprocessor)
    ])

    return feature_engineering_pipeline

# --- Main Execution Block (for testing or direct use) ---

def main():
    """
    Main function to load data, perform feature engineering, and display results.
    This block is for demonstration and testing the script's functionality.
    In a real project, this might be called from 'train.py' or other scripts.
    """
    print("Starting feature engineering process...")

    # Define paths relative to the project root for data loading
    # Assuming this script is in src/, the project root is one level up from src/
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir)) # Corrected project_root calculation

    # Corrected: Data is directly in 'data/' folder
    data_path = os.path.join(project_root, 'data', 'data.csv')
    variable_definitions_path = os.path.join(project_root, 'data', 'Xente_Variable_Definitions.csv')

    try:
        df_raw = pd.read_csv(data_path)
        df_var_definitions = pd.read_csv(variable_definitions_path)
        print(f"Raw data loaded successfully from: {data_path}")
        print(f"Variable definitions loaded successfully from: {variable_definitions_path}")
        print(f"Raw data shape: {df_raw.shape}")
    except FileNotFoundError:
        print(f"Error: Data file(s) not found.")
        print(f"Please ensure 'data.csv' and 'Xente_Variable_Definitions.csv' are in your project's 'data/' directory.")
        print(f"Attempted path for data.csv: {data_path}")
        print(f"Attempted path for Xente_Variable_Definitions.csv: {variable_definitions_path}")
        return

    # --- Define Feature Lists ---
    # These lists should be carefully chosen based on your EDA findings from Task 2.
    # Include both original numerical columns and the names of the new features that will be created.
    # Note: 'Amount' and 'Value' are often highly correlated. You might choose one or use both.
    # For now, let's include both and let the model decide.
    
    # Original numerical features (excluding IDs that are not measures like TransactionId, CustomerId etc.)
    original_numerical_cols = ['Amount', 'Value']

    # New numerical features created by FeatureExtractor and CustomerAggregator
    extracted_time_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    aggregated_customer_features = [
        'TotalTransactionAmount', 'AverageTransactionAmount',
        'TransactionCount', 'StdDevTransactionAmounts'
    ]

    # Combine all numerical features that will go into the numerical pipeline
    all_numerical_features = original_numerical_cols + extracted_time_features + aggregated_customer_features

    # Categorical features to be One-Hot Encoded
    # CountryCode is numerical but represents a category, so it's treated as categorical here.
    categorical_features_onehot = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]

    # --- Create and Apply the Pipeline ---
    feature_pipeline = create_feature_engineering_pipeline(
        numerical_features=all_numerical_features,
        categorical_features_onehot=categorical_features_onehot
    )

    # Fit and transform the data using the pipeline
    # The pipeline will handle:
    # 1. Extracting time features
    # 2. Aggregating customer features
    # 3. Imputing missing values (if any)
    # 4. Encoding categorical variables
    # 5. Scaling numerical features
    transformed_data_array = feature_pipeline.fit_transform(df_raw)

    # --- CORRECTED: Get feature names after transformation for the final DataFrame ---
    # The get_feature_names_out() method of ColumnTransformer is the most reliable way.
    # It requires the pipeline to be fitted first.

    # Access the preprocessor step which is the ColumnTransformer within the feature_pipeline
    preprocessor_transformer = feature_pipeline.named_steps['preprocessor']

    # Get all output feature names from the ColumnTransformer
    # This will automatically include names from 'num', 'cat_onehot', and 'remainder'
    all_transformed_feature_names = preprocessor_transformer.get_feature_names_out()

    # Create the final transformed DataFrame
    df_processed = pd.DataFrame(transformed_data_array, columns=all_transformed_feature_names)

    print("\nFeature Engineering completed!")
    print(f"Processed data shape: {df_processed.shape}")
    print("\nFirst 5 rows of the processed data:")
    print(df_processed.head())
    print("\nProcessed data info:")
    df_processed.info()

    # You can optionally save the processed data for later use (e.g., in data/processed/)
    # Create the 'data/processed' directory if it doesn't exist
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    processed_data_output_path = os.path.join(processed_dir, 'processed_transactions.csv')
    df_processed.to_csv(processed_data_output_path, index=False)
    print(f"\nProcessed data saved to: {processed_data_output_path}")


if __name__ == "__main__":
    main()