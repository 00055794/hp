"""
Corrected Neural Network Training Script for Kazakhstan House Price Prediction
Based on 3_GZ_House_Prices_Model_fromMLOps.ipynb (updated for NEW 13-feature model)

This script:
1. Loads hp_segments_points.parquet with region-enriched house price data
2. Prepares features matching the web app: ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, 
   FLOOR, TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, MATERIAL, YEAR, 
   REGION_GRID, segment_code
3. Trains a PyTorch neural network with early stopping
4. Exports model, scalers, feature_list, and cat_mappings for the web app

NEW FEATURES:
- REGION_GRID: Integer-encoded region from lat/lon grid lookup (replaces CITY_CAT)
- LONGITUDE, LATITUDE: Direct coordinate inputs
- segment_code: Market segment code (int32)
REMOVED: CITY_CAT, srednmes_zarplata (salary data no longer used)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump


# ==================== MODEL DEFINITION ====================
class HousePriceModel(nn.Module):
    """
    PyTorch neural network for house price prediction
    Architecture: Input -> 64 -> 16 -> 1 (regression output)
    """
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


# ==================== DATA PREPROCESSING ====================
def load_and_preprocess_data(data_path: str):
    """
    Load and preprocess house price data from the notebook output.
    
    Expected columns in hp_segments_points.parquet (NEW 13-feature model):
    - PRICE_ln: Target variable (log-transformed price)
    - ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR, TOTAL_FLOORS: Numeric features
    - FURNITURE, CONDITION, CEILING, MATERIAL, YEAR: Encoded numeric features
    - REGION_GRID: Integer-encoded region from grid lookup (int32)
    - segment_code: Market segment identifier (int32)
    
    REMOVED FEATURES (no longer used):
    - CITY_CAT, srednmes_zarplata (replaced by REGION_GRID)
    - segment_id (use segment_code directly)
    - geometry, index_right, id, etc.: Columns to drop
    """
    print(f"Loading data from: {data_path}")
    
    # Load the data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.geojson'):
        df = gpd.read_file(data_path)
        if 'geometry' in df.columns:
            # Convert to regular DataFrame
            df = pd.DataFrame(df.drop(columns=['geometry']))
    else:
        raise ValueError(f"Expected .parquet or .geojson, got: {data_path}")
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Encode segment_id -> segment_code
    if 'segment_id' in df.columns and 'segment_code' not in df.columns:
        le = LabelEncoder()
        df['segment_code'] = le.fit_transform(df['segment_id'].astype(str))
        print(f"Encoded segment_id -> segment_code: {df['segment_code'].nunique()} unique segments")
    
    # Drop columns that are not features (KEEP LONGITUDE, LATITUDE for NEW model)
    drop_cols = []
    for col in ['geometry', 'index_right', 'id', 
                'CITY_CAT', 'srednmes_zarplata',  # OLD features no longer used
                'chislennost_naseleniya_092025',
                'temp_prirosta_percent',
                'house_point_count', 'area_km2', 'density_hp_per_km2',
                'f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7',
                'segment_local', 'segment_id']:  # drop segment_id, keep segment_code
        if col in df.columns:
            drop_cols.append(col)
    
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped {len(drop_cols)} columns: {drop_cols}")
    
    # Normalize column names (handle 'TOTAL AREA' with space)
    if 'TOTAL AREA' in df.columns:
        df.rename(columns={'TOTAL AREA': 'TOTAL_AREA'}, inplace=True)
        print("Renamed 'TOTAL AREA' -> 'TOTAL_AREA'")
    
    # Ensure target column exists
    if 'PRICE_ln' not in df.columns:
        raise ValueError("Target column 'PRICE_ln' not found!")
    
    # Check for required features (NEW 13-feature model)
    expected_features = [
        'ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL_AREA', 'FLOOR', 'TOTAL_FLOORS',
        'FURNITURE', 'CONDITION', 'CEILING', 'MATERIAL', 'YEAR',
        'REGION_GRID', 'segment_code'
    ]
    
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        print(f"WARNING: Missing expected features: {missing}")
        print(f"Available columns: {list(df.columns)}")
    
    # Handle NaN values
    print("\nChecking for missing values...")
    na_counts = df.isna().sum()
    if na_counts.any():
        print("Missing values per column:")
        print(na_counts[na_counts > 0])
        
        # Fill missing values with median/mode
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['float64', 'int64']:
                    fill_val = df[col].median()
                    df[col].fillna(fill_val, inplace=True)
                    print(f"Filled {col} NaNs with median: {fill_val}")
                else:
                    fill_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df[col].fillna(fill_val, inplace=True)
                    print(f"Filled {col} NaNs with mode: {fill_val}")
    
    print(f"\nFinal data shape: {df.shape}")
    return df


# ==================== TRAINING FUNCTION ====================
def train_model(data_path: str, out_dir: str, epochs: int = 100, 
                batch_size: int = 32, lr: float = 0.001,
                test_size: float = 0.3, random_state: int = 0):
    """
    Train the neural network model
    """
    # Set seeds for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data(data_path)
    
    # Separate features and target
    target_column = 'PRICE_ln'
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found!")
    
    features = data.drop(columns=[target_column])
    target = data[target_column]
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Save feature list for inference
    feature_list = features.columns.tolist()
    feature_list_path = os.path.join(out_dir, "feature_list.json")
    with open(feature_list_path, "w", encoding="utf-8") as f:
        json.dump(feature_list, f, ensure_ascii=False, indent=2)
    print(f"Saved feature list to: {feature_list_path}")
    
    # Split the data
    print(f"\nSplitting data: train={1-test_size:.0%}, valid={test_size/2:.0%}, test={test_size/2:.0%}")
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    print(f"Train: {len(X_train_full)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    # Scale features and target using MinMaxScaler
    print("\nScaling features and target...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_full)
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(np.array(y_train_full).reshape(-1, 1))
    y_valid_scaled = scaler_y.transform(np.array(y_valid).reshape(-1, 1))
    y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1, 1))
    
    # Check for NaNs/Infs
    assert not np.isnan(X_train_scaled).any(), "NaNs in X_train"
    assert not np.isnan(y_train_scaled).any(), "NaNs in y_train"
    assert not np.isinf(X_train_scaled).any(), "Infs in X_train"
    assert not np.isinf(y_train_scaled).any(), "Infs in y_train"
    
    # Convert to PyTorch tensors
    print("Converting to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_scaled, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)
    
    # Initialize model
    input_dim = X_train_scaled.shape[1]
    print(f"\nInitializing model with input_dim={input_dim}")
    model = HousePriceModel(input_dim)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Early stopping
    early_stopping_patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("="*70)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_valid_tensor)
            val_loss = criterion(val_outputs, y_valid_tensor).item()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Best Val: {best_val_loss:.6f}, "
                  f"No Improve: {epochs_no_improve}")
        
        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (val_loss={best_val_loss:.6f})")
    
    # ==================== EVALUATION ====================
    print("\n" + "="*70)
    print("Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_test_pred_scaled = model(X_test_tensor).numpy()
    
    # Inverse transform predictions and targets
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    y_test_original = scaler_y.inverse_transform(y_test_scaled).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_original, y_test_pred)
    mse = mean_squared_error(y_test_original, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_test_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_original - y_test_pred) / y_test_original)) * 100
    
    print(f"\nTest Set Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # ==================== SAVE ARTIFACTS ====================
    print("\n" + "="*70)
    print("Saving model artifacts...")
    
    # Save model
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved model to: {model_path}")
    
    # Save scalers
    scaler_X_path = os.path.join(out_dir, "scaler_X.joblib")
    scaler_y_path = os.path.join(out_dir, "scaler_y.joblib")
    dump(scaler_X, scaler_X_path)
    dump(scaler_y, scaler_y_path)
    print(f"✓ Saved scaler_X to: {scaler_X_path}")
    print(f"✓ Saved scaler_y to: {scaler_y_path}")
    
    # Save metadata
    metadata = {
        "input_dim": input_dim,
        "feature_names": feature_list,
        "target_column": target_column,
        "test_metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape)
        },
        "training_config": {
            "epochs": epoch + 1,
            "batch_size": batch_size,
            "learning_rate": lr,
            "early_stopping_patience": early_stopping_patience,
            "random_state": random_state
        }
    }
    
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # Create empty cat_mappings.json (no categorical encoding needed)
    cat_mappings_path = os.path.join(out_dir, "cat_mappings.json")
    with open(cat_mappings_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved cat_mappings to: {cat_mappings_path}")
    
    print("\n" + "="*70)
    print("✅ Training completed successfully!")
    print(f"All artifacts saved to: {out_dir}")
    
    return model, scaler_X, scaler_y, metadata


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house price prediction neural network")
    parser.add_argument("--data", type=str, 
                        default="../data2/hp_segments_points.parquet",
                        help="Path to input data file (.parquet or .geojson)")
    parser.add_argument("--output", type=str, 
                        default="./nn_model",
                        help="Output directory for model artifacts")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Test+validation set size (will be split 50/50)")
    parser.add_argument("--random_state", type=int, default=0,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run training
    train_model(
        data_path=args.data,
        out_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_size=args.test_size,
        random_state=args.random_state
    )
