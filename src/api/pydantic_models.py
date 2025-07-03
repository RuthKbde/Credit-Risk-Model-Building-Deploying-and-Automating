# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import List, Optional

class TransactionData(BaseModel):
    # These should be the RAW features your FeatureExtractor and CustomerAggregator initially receive
    # and any other raw features that go into your final pipeline before encoding/scaling.
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float # Raw amount can be float or str, assume float after initial cleaning
    Value: float  # Raw value can be float or str, assume float after initial cleaning
    TransactionStartTime: str # Keep as string, FeatureExtractor will parse
    PricingStrategy: int
    # FraudResult is not needed for prediction input

class PredictionResult(BaseModel):
    # This is the output model for your prediction
    # Assuming your model outputs a probability (float between 0 and 1)
    risk_probability: float
    # You might also want to include a risk label (e.g., "low_risk", "high_risk")
    # risk_label: str