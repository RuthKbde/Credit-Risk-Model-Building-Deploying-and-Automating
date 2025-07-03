import pytest
import pandas as pd
import numpy as np
from src.data_processing import FeatureExtractor, CustomerAggregator # Import your custom transformers

# --- Test Data ---
# Create a small, representative dummy DataFrame for testing
@pytest.fixture
def sample_data():
    data = {
        'TransactionId': [1, 2, 3, 4, 5, 6, 7, 8],
        'BatchId': [101, 101, 102, 102, 103, 103, 104, 104],
        'AccountId': [1001, 1001, 1002, 1002, 1001, 1003, 1003, 1004],
        'SubscriptionId': [201, 201, 202, 202, 201, 203, 203, 204],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C1', 'C3', 'C3', 'C4'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX', 'USD', 'UGX', 'UGX', 'UGX'],
        'CountryCode': [256, 256, 256, 256, 1, 256, 256, 256],
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P1', 'P2', 'P1', 'P3'],
        'ProductId': ['ProdA', 'ProdB', 'ProdA', 'ProdC', 'ProdA', 'ProdB', 'ProdC', 'ProdA'],
        'ProductCategory': ['CatA', 'CatB', 'CatA', 'CatC', 'CatA', 'CatB', 'CatC', 'CatA'],
        'ChannelId': ['Web', 'App', 'Web', 'App', 'Web', 'App', 'Web', 'App'],
        'Amount': [100.0, 50.0, 200.0, 75.0, 150.0, 300.0, 100.0, 500.0],
        'Value': [100.0, 50.0, 200.0, 75.0, 150.0, 300.0, 100.0, 500.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00',
            '2023-01-01 11:30:00',
            '2023-01-02 14:00:00',
            '2023-01-02 15:00:00',
            '2023-01-05 09:00:00', # C1's latest transaction
            '2023-01-03 18:00:00',
            '2023-01-03 19:00:00',
            '2023-01-04 20:00:00'
        ],
        'PricingStrategy': [0, 1, 0, 2, 0, 1, 0, 2],
        'FraudResult': [0, 0, 0, 0, 0, 1, 0, 0] # Example fraud
    }
    return pd.DataFrame(data)

# --- Unit Tests for FeatureExtractor ---

def test_feature_extractor_columns(sample_data):
    """Test if FeatureExtractor adds the correct new columns."""
    extractor = FeatureExtractor()
    transformed_df = extractor.transform(sample_data.copy()) # Use a copy to not modify fixture

    expected_cols = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    for col in expected_cols:
        assert col in transformed_df.columns, f"Column {col} not found after transformation."

def test_feature_extractor_values(sample_data):
    """Test if extracted feature values are correct for a known transaction."""
    extractor = FeatureExtractor()
    transformed_df = extractor.transform(sample_data.copy())

    # Test the first row's extracted values
    # Original: '2023-01-01 10:00:00'
    assert transformed_df.loc[0, 'TransactionHour'] == 10
    assert transformed_df.loc[0, 'TransactionDay'] == 1
    assert transformed_df.loc[0, 'TransactionMonth'] == 1
    assert transformed_df.loc[0, 'TransactionYear'] == 2023

    # Test a different row (e.g., row 4: '2023-01-05 09:00:00')
    assert transformed_df.loc[4, 'TransactionHour'] == 9
    assert transformed_df.loc[4, 'TransactionDay'] == 5
    assert transformed_df.loc[4, 'TransactionMonth'] == 1
    assert transformed_df.loc[4, 'TransactionYear'] == 2023

def test_feature_extractor_nan_handling(sample_data):
    """Test FeatureExtractor's handling of invalid date strings."""
    # Create data with an invalid date string
    data_with_nan = sample_data.copy()
    data_with_nan.loc[1, 'TransactionStartTime'] = 'invalid-date' # Introduce an invalid date

    extractor = FeatureExtractor()
    transformed_df = extractor.transform(data_with_nan)

    # Check if invalid date resulted in -1 for extracted features at index 1
    assert transformed_df.loc[1, 'TransactionHour'] == -1
    assert transformed_df.loc[1, 'TransactionDay'] == -1
    assert transformed_df.loc[1, 'TransactionMonth'] == -1
    assert transformed_df.loc[1, 'TransactionYear'] == -1


# --- Unit Tests for CustomerAggregator ---

def test_customer_aggregator_columns(sample_data):
    """Test if CustomerAggregator adds the correct aggregate columns."""
    aggregator = CustomerAggregator()
    transformed_df = aggregator.transform(sample_data.copy())

    expected_cols = [
        'TotalTransactionAmount', 'AverageTransactionAmount',
        'TransactionCount', 'StdDevTransactionAmounts'
    ]
    for col in expected_cols:
        assert col in transformed_df.columns, f"Aggregate column {col} not found."

def test_customer_aggregator_values(sample_data):
    """Test if aggregate values are correctly calculated for specific customers."""
    aggregator = CustomerAggregator()
    transformed_df = aggregator.transform(sample_data.copy())

    # Test Customer 'C1'
    # Transactions: 100.0 (idx 0), 50.0 (idx 1), 150.0 (idx 4)
    # Total: 300.0, Count: 3, Avg: 100.0, Std: np.std([100, 50, 150], ddof=1) = 50.0
    c1_data = transformed_df[transformed_df['CustomerId'] == 'C1'].iloc[0]
    assert c1_data['TotalTransactionAmount'] == 300.0
    assert c1_data['AverageTransactionAmount'] == 100.0
    assert c1_data['TransactionCount'] == 3
    assert np.isclose(c1_data['StdDevTransactionAmounts'], 50.0) # Using np.isclose for float comparison

    # Test Customer 'C2'
    # Transactions: 200.0 (idx 2), 75.0 (idx 3)
    # Total: 275.0, Count: 2, Avg: 137.5, Std: np.std([200, 75], ddof=1) = 88.38834764831843
    c2_data = transformed_df[transformed_df['CustomerId'] == 'C2'].iloc[0]
    assert c2_data['TotalTransactionAmount'] == 275.0
    assert c2_data['AverageTransactionAmount'] == 137.5
    assert c2_data['TransactionCount'] == 2
    assert np.isclose(c2_data['StdDevTransactionAmounts'], np.std([200, 75], ddof=1))

    # Test Customer 'C4' (single transaction)
    # Transactions: 500.0 (idx 7)
    # Total: 500.0, Count: 1, Avg: 500.0, Std: 0.0 (as per fillna(0) logic in transformer)
    c4_data = transformed_df[transformed_df['CustomerId'] == 'C4'].iloc[0]
    assert c4_data['TotalTransactionAmount'] == 500.0
    assert c4_data['AverageTransactionAmount'] == 500.0
    assert c4_data['TransactionCount'] == 1
    assert c4_data['StdDevTransactionAmounts'] == 0.0 # Should be 0 for single transaction after fillna