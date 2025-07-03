import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import datetime # To define the snapshot date

print("Script started: create_target_variable.py")

# --- Configuration and Constants ---
# Define paths relative to the project root
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed', 'processed_transactions.csv')
OUTPUT_DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'processed', 'processed_data_with_target.csv')

# K-Means clustering parameters
N_CLUSTERS = 3
RANDOM_STATE = 42 # For reproducibility

# --- Main Function for Target Variable Engineering ---

def create_proxy_target_variable():
    """
    Orchestrates the creation of a proxy target variable ('is_high_risk')
    based on RFM analysis and K-Means clustering.
    """
    print("Starting proxy target variable engineering...")

    # 1. Load Processed Data
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Processed data loaded successfully from: {PROCESSED_DATA_PATH}")
        print(f"Processed data shape: {df.shape}")
        
        # Ensure TransactionStartTime is datetime if it was passed through as object
        if 'remainder__TransactionStartTime' in df.columns:
            df['remainder__TransactionStartTime'] = pd.to_datetime(df['remainder__TransactionStartTime'], errors='coerce')
        else: # Fallback if column naming convention changes or if it was handled differently
             print("Warning: 'remainder__TransactionStartTime' not found. Assuming 'TransactionStartTime' or similar is present and is datetime for RFM.")
             # You might need to adjust this based on your exact column name if 'remainder__' prefix isn't there
             if 'TransactionStartTime' in df.columns:
                 df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
             else:
                 raise ValueError("TransactionStartTime column not found in processed data.")

    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}.")
        print("Please ensure feature engineering (Task 3) has been completed and data saved.")
        return

    # Identify the correct CustomerId and Amount columns after feature engineering
    # They should have the 'remainder__' prefix if they were passed through
    customer_id_col = 'remainder__CustomerId'
    # The 'Amount' column was scaled to 'num__Amount'. For RFM Monetary calculation,
    # we need the original 'Amount' or handle scaling in a way that allows sum of original.
    # A safer way for RFM is to use the original 'Amount' before scaling.
    # If the original 'Amount' is no longer directly available (e.g., only 'num__Amount' exists),
    # you might need to reconsider passing it through or reverse transform 'num__Amount'.
    # For now, let's assume 'Amount' was kept via remainder or we revert scaling for RFM.
    # A more robust solution for RFM: perform RFM *before* full scaling, then merge.
    # However, to stick to the sequence of using 'processed_transactions.csv',
    # we need to ensure the raw 'Amount' is available or assume 'num__Amount' can be used after inverse scaling.
    # Given the previous `df.info()` output, it seems `Amount` and `Value` are still numerical despite 'object' dtype.
    # Let's try to convert them back to numeric for RFM calculation.
    
    # We need the original, unscaled amount for meaningful monetary calculations.
    # If 'Amount' was only scaled and not passed through directly, it's safer to use the original raw 'Amount' column.
    # A better practice would be to ensure 'Amount' is kept as 'passthrough' for RFM if you're using the processed data directly.
    # For this exercise, let's assume the original 'Amount' and 'TransactionStartTime'
    # were kept (or we can extract them if possible).
    # Since they were included in 'num__', they are scaled.
    # A direct sum of scaled values might not make sense for 'Monetary'.

    # REVISED RFM STRATEGY:
    # It's generally better to calculate RFM *before* scaling or on the original raw data,
    # then merge the RFM features and target *into* the processed data.
    # Let's adjust this for robust RFM calculation.
    # For this, we temporarily load the raw data again just for RFM.
    print("Loading raw data for RFM calculation (to use original Amount and TransactionStartTime)...")
    raw_data_path_for_rfm = os.path.join(PROJECT_ROOT, 'data', 'data.csv')
    try:
        df_raw_for_rfm = pd.read_csv(raw_data_path_for_rfm)
        df_raw_for_rfm['TransactionStartTime'] = pd.to_datetime(df_raw_for_rfm['TransactionStartTime'], errors='coerce')
        # Ensure 'Amount' is numeric
        df_raw_for_rfm['Amount'] = pd.to_numeric(df_raw_for_rfm['Amount'], errors='coerce').fillna(0)
        
        # Drop rows where TransactionStartTime is NaT after coercion, as they can't be used for Recency
        df_raw_for_rfm.dropna(subset=['TransactionStartTime'], inplace=True)
        print(f"Raw data for RFM loaded successfully. Shape: {df_raw_for_rfm.shape}")
    except FileNotFoundError:
        print(f"Error: Raw data file for RFM not found at {raw_data_path_for_rfm}.")
        return

    # 2. Calculate RFM Metrics
    # Define a snapshot date: A date slightly after the latest transaction in your dataset.
    # This ensures all transactions are included and recency is consistent.
    snapshot_date = df_raw_for_rfm['TransactionStartTime'].max() + pd.Timedelta(days=1)
    print(f"Using snapshot date for RFM calculation: {snapshot_date}")

    rfm_df = df_raw_for_rfm.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()

    print(f"RFM DataFrame created. Shape: {rfm_df.shape}")
    print("RFM Head:\n", rfm_df.head())

    # Handle potential zero or negative monetary values (if applicable to your domain)
    # E.g., for credit risk, transactions might mostly be positive. If negative values exist,
    # consider taking absolute sum or handle them as a separate risk factor.
    # For now, we assume positive amounts for 'Monetary'.

    # It's crucial to handle cases where Monetary might be 0 due to 0-amount transactions
    # or if Amount was all NaN and filled with 0.
    # If a customer has 0 frequency, Monetary will also be 0.
    # Recency for a customer with no transactions would be max_days if not dropped,
    # but groupby 'count' will exclude non-existent customers.

    # 3. Pre-process RFM Features
    rfm_features = ['Recency', 'Frequency', 'Monetary']
    
    # Handle infinite values (e.g., if Recency max - min results in huge number)
    rfm_df[rfm_features] = rfm_df[rfm_features].replace([np.inf, -np.inf], np.nan)
    
    # Impute any NaNs that might have been introduced (e.g. if a customer had no valid transactions after all filtering)
    # This is less likely after initial dropna, but good for robustness.
    # For Recency, median might be appropriate. For Frequency/Monetary, 0 might be.
    # Let's use SimpleImputer inside a pipeline for consistency with sklearn.
    
    rfm_scaler = StandardScaler()
    rfm_scaled = rfm_scaler.fit_transform(rfm_df[rfm_features])
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_features, index=rfm_df.index)

    print("RFM features scaled.")
    print("Scaled RFM Head:\n", rfm_scaled_df.head())

    # 4. Cluster Customers
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10) # n_init for modern KMeans
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

    print(f"Customers clustered into {N_CLUSTERS} groups.")
    print("Cluster distribution:\n", rfm_df['Cluster'].value_counts())

    # Analyze Cluster Centroids to identify the 'High-Risk' group
    # We want the cluster with high Recency (long time since last purchase), low Frequency, low Monetary.
    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=rfm_features)
    print("\nCluster Centroids (Scaled Values):")
    print(cluster_centroids)

    # To interpret, you can inverse transform the centroids:
    cluster_centroids_unscaled = pd.DataFrame(rfm_scaler.inverse_transform(kmeans.cluster_centers_), columns=rfm_features)
    print("\nCluster Centroids (Original Scale):")
    print(cluster_centroids_unscaled)

    # Identify the high-risk cluster
    # Look for the cluster with the highest 'Recency' and lowest 'Frequency' and 'Monetary'.
    # This often means sorting by Recency (desc) and Frequency (asc) etc.
    # Example logic:
    # Assuming lower Frequency and Monetary, and higher Recency indicate less engagement.
    # We'll pick the cluster with the highest mean Recency and lowest mean Frequency/Monetary
    
    # Sort centroids to find the "worst" cluster
    # Higher Recency is worse, lower Frequency is worse, lower Monetary is worse.
    # Normalize these to score and sum, or pick based on clear patterns.
    # For now, let's look at the unscaled centroids and assume the one with highest Recency and lowest F/M is the target.
    
    # A simple heuristic: Sort by Recency (desc), then Frequency (asc), then Monetary (asc)
    # The first row after sorting might be our high-risk cluster.
    
    # A more robust way: create a combined "risk score" for each cluster based on scaled centroids
    # For example, score = scaled_recency - scaled_frequency - scaled_monetary
    # (High scaled_recency, low scaled_frequency/monetary contribute to higher risk score)
    cluster_centroids_unscaled['Risk_Score'] = (
        cluster_centroids_unscaled['Recency'] / cluster_centroids_unscaled['Recency'].max() + # higher is worse
        (1 - cluster_centroids_unscaled['Frequency'] / cluster_centroids_unscaled['Frequency'].max()) + # lower is worse
        (1 - cluster_centroids_unscaled['Monetary'] / cluster_centroids_unscaled['Monetary'].max()) # lower is worse
    )
    
    # Or, simpler: high recency, low frequency, low monetary
    # Find the cluster with max scaled Recency and min scaled Frequency/Monetary
    
    # Let's identify by inspecting the unscaled centroids for now as it's more interpretable.
    # Assuming cluster 0 has high Recency, low F, low M (e.g., from visual inspection of output)
    # YOU WILL NEED TO ADJUST 'HIGH_RISK_CLUSTER_ID' AFTER INSPECTING YOUR OUTPUT.
    # Example: If cluster 0 has unscaled centroids like (Recency: 300, Frequency: 1, Monetary: 50)
    # and other clusters are (Recency: 10, Frequency: 20, Monetary: 1000) and (Recency: 50, Frequency: 5, Monetary: 200)
    # then cluster 0 would be the high-risk one.

    # Placeholder for the high-risk cluster ID: YOU MUST DETERMINE THIS FROM THE CENTROID OUTPUT
    # For example, if cluster_centroids_unscaled shows cluster 2 has highest Recency and lowest F/M, set high_risk_cluster_id = 2
    # BASED ON COMMON RFM PATTERNS, THE CLUSTER WITH THE HIGHEST RECENCY, LOWEST FREQUENCY, AND LOWEST MONETARY VALUE IS THE 'LEAST ENGAGED'.
    
    # Let's try to determine it programmatically:
    # Create a normalized "badness" score for each cluster
    # Example: higher Recency is worse, lower Frequency is worse, lower Monetary is worse.
    # For each cluster, we can sum the scaled values in a way that points to 'bad'.
    # If scaled Recency is high, scaled Frequency is low, scaled Monetary is low.
    
    # The most common way to identify the "least engaged" cluster from scaled centroids is to find the one
    # that is furthest in the direction of high Recency, low Frequency, low Monetary.
    # Let's find the cluster where:
    #   Scaled Recency is highest
    #   Scaled Frequency is lowest
    #   Scaled Monetary is lowest
    
    # This might require a manual decision, but we can attempt a programmatic one:
    # For simplicity, let's sort by scaled Recency (desc), then scaled Frequency (asc), then scaled Monetary (asc)
    # This doesn't guarantee the "least engaged" but points to it.
    
    # A more robust programmatic way: find the cluster whose centroid is furthest from the origin (0,0,0) in terms of badness.
    # Badness = Scaled_Recency - Scaled_Frequency - Scaled_Monetary
    
    # For now, I'll put a placeholder and you'll confirm after running.
    # After inspecting the printed `cluster_centroids_unscaled` output,
    # manually identify the cluster ID that represents "high Recency, low Frequency, low Monetary".
    # For example, if cluster_centroids_unscaled shows row index 0 has the desired characteristics:
    # high_risk_cluster_id = 0
    
    # **** IMPORTANT: YOU NEED TO MANUALLY INSPECT THE 'Cluster Centroids (Original Scale)' OUTPUT *****
    # **** AND SET THE high_risk_cluster_id ACCORDINGLY. ****
    # Example: If Cluster 0 has Recency ~300 days, Freq ~1, Monetary ~50 (very low engagement)
    #          And other clusters have much better values, then set high_risk_cluster_id = 0
    high_risk_cluster_id = None # Placeholder. You MUST set this after inspection.
    
    # Example of how you might programmatically select it if the pattern is clear:
    # Find the cluster with the highest mean Recency
    high_recency_cluster_id = cluster_centroids_unscaled['Recency'].idxmax()
    # Find the cluster with the lowest mean Frequency
    low_frequency_cluster_id = cluster_centroids_unscaled['Frequency'].idxmin()
    # Find the cluster with the lowest mean Monetary
    low_monetary_cluster_id = cluster_centroids_unscaled['Monetary'].idxmin()

    # If these three all point to the same cluster, that's your high-risk one.
    # If not, you need to use your domain knowledge from EDA.
    
    # For the purpose of this script, let's assume we choose the cluster that has the highest scaled Recency
    # and the lowest scaled Frequency/Monetary.
    # A simple approach: sort by Recency (descending), then by Frequency (ascending), then Monetary (ascending)
    # This heuristic often points to the 'worst' cluster first.
    sorted_centroids = cluster_centroids_unscaled.sort_values(
        by=['Recency', 'Frequency', 'Monetary'],
        ascending=[False, True, True]
    )
    high_risk_cluster_id = sorted_centroids.index[0] # Pick the first one after sorting by "worst" criteria
    print(f"\nProgrammatically identified potential high-risk cluster (based on Recency, Freq, Monetary sort): {high_risk_cluster_id}")
    print(f"Its unscaled centroid is:\n{cluster_centroids_unscaled.loc[high_risk_cluster_id]}")
    # Please manually verify this choice after running the script!

    # 5. Define and Assign the "High-Risk" Label
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)
    print(f"\nAssigned 'is_high_risk' label. High-risk count: {rfm_df['is_high_risk'].sum()}")
    print(f"Proportion of high-risk customers: {rfm_df['is_high_risk'].mean():.2%}")

    # 6. Integrate the Target Variable
    # Merge the 'is_high_risk' column back into the main processed DataFrame
    # Use the CustomerId column for merging. The processed DF has 'remainder__CustomerId'.
    
    # Ensure CustomerId column in rfm_df matches the one in df for merging
    if 'CustomerId' in rfm_df.columns:
        # Assuming df has 'remainder__CustomerId' as the CustomerId column from `remainder='passthrough'`
        df_merged = pd.merge(df, rfm_df[['CustomerId', 'is_high_risk']],
                             left_on=customer_id_col, right_on='CustomerId', how='left')
        df_merged.drop(columns='CustomerId', inplace=True) # Drop the redundant CustomerId column from rfm_df
    else:
        print("Warning: 'CustomerId' not found in RFM DataFrame. Merge might fail or be incorrect.")
        df_merged = df # Fallback, but merge logic needs validation

    # Handle customers in df who might not be in rfm_df (e.g., no valid transactions for RFM)
    # Assign 0 (low-risk) by default if no RFM data was found for them.
    df_merged['is_high_risk'] = df_merged['is_high_risk'].fillna(0).astype(int)

    print(f"\nTarget variable 'is_high_risk' merged into main dataset. Shape: {df_merged.shape}")
    print("First 5 rows of merged data with target:")
    print(df_merged.head())
    print("\nValue counts for 'is_high_risk':")
    print(df_merged['is_high_risk'].value_counts())

    # 7. Save Output
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    df_merged.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\nProcessed data with target saved to: {OUTPUT_DATA_PATH}")
    
    return df_merged # Return the DataFrame for potential chained operations or testing

if __name__ == "__main__":
    create_proxy_target_variable()