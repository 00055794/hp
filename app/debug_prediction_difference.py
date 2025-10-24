"""
Debug script to compare notebook vs web service predictions
"""
import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from nn_inference import load_artifacts, apply_cat_maps, build_feature_matrix

def compare_predictions():
    """Compare predictions on training data sample"""
    print("="*80)
    print("DEBUGGING PREDICTION DIFFERENCE")
    print("="*80)
    
    # Load model artifacts
    print("\n1. Loading model artifacts...")
    artifacts = load_artifacts("nn_model")
    print(f"   Feature list: {artifacts.feature_list}")
    
    # Load training data
    print("\n2. Loading training data...")
    data = pd.read_parquet("hp_segments_points.parquet")
    print(f"   Data shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Take a few samples
    samples = data.sample(5, random_state=42)
    
    print("\n3. Making predictions on 5 random samples...")
    print("="*80)
    
    for idx, (i, row) in enumerate(samples.iterrows(), 1):
        actual_price_ln = row['PRICE_ln']
        actual_price_kzt = np.exp(actual_price_ln)
        
        # Create input DataFrame matching web service format
        input_df = pd.DataFrame([{
            'ROOMS': row['ROOMS'],
            'LONGITUDE': row['LONGITUDE'],
            'LATITUDE': row['LATITUDE'],
            'TOTAL_AREA': row['TOTAL_AREA'],
            'FLOOR': row['FLOOR'],
            'TOTAL_FLOORS': row['TOTAL_FLOORS'],
            'FURNITURE': row['FURNITURE'],
            'CONDITION': row['CONDITION'],
            'CEILING': row['CEILING'],
            'MATERIAL': row['MATERIAL'],
            'YEAR': row['YEAR'],
            'REGION_GRID': row['REGION_GRID'],
            'segment_code': row['segment_code']
        }])
        
        # Apply categorical mappings
        df_enc = apply_cat_maps(input_df.copy(), artifacts.cat_maps)
        
        # Build feature matrix
        X = build_feature_matrix(df_enc, artifacts.feature_list)
        
        # Scale
        X_scaled = artifacts.scaler_X.transform(X)
        
        # Predict
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_scaled = artifacts.model(x_tensor).numpy()
        
        # Inverse transform
        y_ln = artifacts.scaler_y.inverse_transform(y_scaled).flatten()[0]
        pred_price_kzt = np.exp(y_ln)
        
        error_pct = ((pred_price_kzt - actual_price_kzt) / actual_price_kzt) * 100
        
        print(f"\nSample {idx}:")
        print(f"  Features: Rooms={row['ROOMS']}, Area={row['TOTAL_AREA']}m², "
              f"Floor={row['FLOOR']}/{row['TOTAL_FLOORS']}, Year={row['YEAR']}")
        print(f"  REGION_GRID={row['REGION_GRID']}, segment_code={row['segment_code']}")
        print(f"  Actual PRICE_ln: {actual_price_ln:.4f}")
        print(f"  Predicted PRICE_ln: {y_ln:.4f}")
        print(f"  Actual Price:    {actual_price_kzt:>15,.0f} KZT")
        print(f"  Predicted Price: {pred_price_kzt:>15,.0f} KZT")
        print(f"  Error: {error_pct:+.2f}%")
        
        if abs(error_pct) > 5:
            print(f"  ⚠️ WARNING: Error > 5%!")
            print(f"  Feature values in order:")
            for feat_name, feat_val in zip(artifacts.feature_list, X[0]):
                print(f"    {feat_name}: {feat_val}")
            print(f"  Scaled values (first 5): {X_scaled[0][:5]}")
    
    print("\n" + "="*80)
    print("\n4. Checking scaler parameters...")
    print(f"   X scaler min: {artifacts.scaler_X.data_min_}")
    print(f"   X scaler max: {artifacts.scaler_X.data_max_}")
    print(f"   y scaler min: {artifacts.scaler_y.data_min_[0]:.4f} (exp={np.exp(artifacts.scaler_y.data_min_[0]):,.0f} KZT)")
    print(f"   y scaler max: {artifacts.scaler_y.data_max_[0]:.4f} (exp={np.exp(artifacts.scaler_y.data_max_[0]):,.0f} KZT)")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("  If predictions match training data closely (error < 5%):")
    print("    → Web service is working correctly")
    print("    → Check if you're using same REGION_GRID/segment_code in web inputs")
    print("\n  If predictions are consistently 15-20% lower:")
    print("    → Check if REGION_GRID or segment_code differ between notebook and web")
    print("    → Verify categorical encodings (FURNITURE, CONDITION, MATERIAL)")
    print("    → Check scaler parameters match training")
    print("="*80)

if __name__ == "__main__":
    compare_predictions()
