# src/api/main.py
from fastapi import FastAPI, HTTPException
from typing import List
import mlflow
import os
import pandas as pd
from pydantic import ValidationError

# Import your Pydantic models
from .pydantic_models import TransactionData, PredictionResult

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts credit risk probability for new transactions."
)

# --- MLflow Model Loading ---
# IMPORTANT: Replace with your actual MLflow Model URI
# Example: "runs:/your_run_id/model" or "models:/your_model_name/Production"
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/credit_risk_model/Production")

model = None

@app.on_event("startup")
async def load_model():
    """Load the MLflow model when the FastAPI application starts up."""
    global model
    try:
        print(f"Attempting to load MLflow model from: {MLFLOW_MODEL_URI}")
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading MLflow model: {e}")
        # Depending on criticality, you might want to raise the exception to prevent startup
        # or log it and continue with a placeholder/error state.
        # For now, we'll raise it to ensure the API doesn't start without a model.
        raise RuntimeError(f"Failed to load MLflow model: {e}")

# --- Prediction Endpoint ---

@app.post("/predict", response_model=List[PredictionResult])
async def predict_risk(transactions: List[TransactionData]):
    """
    Accepts a list of new customer transaction data and returns risk probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        # Convert incoming list of Pydantic models to a Pandas DataFrame
        # The 'dict' method with 'by_alias=True' handles field aliases if you use them.
        # Ensure the column names in this DataFrame match what your pipeline expects.
        input_df = pd.DataFrame([t.model_dump() for t in transactions])

        # IMPORTANT: If your MLflow model IS NOT a full pipeline that handles
        # FeatureExtractor, CustomerAggregator, encoding, scaling, etc.,
        # you would need to run those preprocessing steps here on `input_df`
        # before passing it to `model.predict` or `model.predict_proba`.
        #
        # For a robust setup, you should save your entire sklearn.pipeline.Pipeline
        # (including all preprocessing steps) with MLflow.
        # If your model variable IS the pipeline, then the call below directly works.

        # Predict probabilities
        # Assuming the model outputs probabilities (e.g., for binary classification)
        # and the positive class probability is at index 1
        predictions_proba = model.predict_proba(input_df)[:, 1].tolist()

        # Format the predictions into the Pydantic PredictionResult model
        results = [{"risk_probability": prob} for prob in predictions_proba]
        
        return results

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Data validation error: {e.errors()}")
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

# Example of a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}