"""
Standalone prediction script that matches the exact notebook pipeline.
This script loads a trained model and makes predictions on feature data.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import os
from pathlib import Path

# Model architecture must match training exactly
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.out_act = nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.out_act(out)
        return out


class HousePricePredictor:
    """Prediction interface matching notebook training pipeline exactly"""
    
    def __init__(self, model_dir="nn_model"):
        """
        Initialize predictor with trained model artifacts.
        
        Args:
            model_dir: Directory containing model.pt, scalers, and feature list
        """
        self.model_dir = Path(model_dir)
        
        # Load feature list (defines order of features)
        with open(self.model_dir / "feature_list.json", "r") as f:
            self.features = json.load(f)
        
        # Load scalers (fitted on training data)
        self.scaler_X = joblib.load(self.model_dir / "scaler_X.joblib")
        self.scaler_y = joblib.load(self.model_dir / "scaler_y.joblib")
        
        # Load model
        input_dim = len(self.features)
        self.model = HousePriceModel(input_dim)
        self.model.load_state_dict(torch.load(self.model_dir / "model.pt", map_location=torch.device('cpu')))
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Features ({input_dim}): {self.features}")
    
    def predict(self, df):
        """
        Make predictions on a DataFrame with the required features.
        
        Args:
            df: pandas DataFrame with columns matching self.features
            
        Returns:
            numpy array of predicted PRICE_ln values (log scale)
        """
        # Validate features
        missing_features = set(self.features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select features in correct order
        X = df[self.features].copy()
        
        # Convert to numeric and handle any object types
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Convert to numpy array with explicit float type
        X = X.astype(float).values
        
        # Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            nan_mask = np.isnan(X) | np.isinf(X)
            nan_cols = [self.features[i] for i in range(len(self.features)) if nan_mask[:, i].any()]
            raise ValueError(f"Input data contains NaN or Inf values in columns: {nan_cols}")
        
        # Scale features (using training scaler)
        X_scaled = self.scaler_X.transform(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Predict (scaled output)
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).numpy()
        
        # Inverse transform to original scale
        y_pred_ln = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        return y_pred_ln
    
    def predict_price(self, df):
        """
        Make predictions and convert from log scale to KZT.
        
        Args:
            df: pandas DataFrame with columns matching self.features
            
        Returns:
            numpy array of predicted prices in KZT
        """
        y_pred_ln = self.predict(df)
        prices_kzt = np.exp(y_pred_ln)
        return prices_kzt


def main():
    """Example usage"""
    # Initialize predictor
    predictor = HousePricePredictor("nn_model")
    
    # Example: Load data and make predictions
    # df = pd.read_parquet("hp_segments_points.parquet")
    # predictions_ln = predictor.predict(df)
    # predictions_kzt = np.exp(predictions_ln)
    
    print("\n✓ Predictor ready. Usage:")
    print("  from predict_from_features import HousePricePredictor")
    print("  predictor = HousePricePredictor('nn_model')")
    print("  prices_kzt = predictor.predict_price(df)")


if __name__ == "__main__":
    main()
