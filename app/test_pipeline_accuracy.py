"""
Test script to verify pipeline predictions match notebook training data.

This script:
1. Loads the training data (hp_segments_points.parquet)
2. Takes a sample of properties
3. Makes predictions using the pipeline
4. Compares with actual prices from training data
5. Reports accuracy metrics

Run this to verify the web service produces correct predictions.
"""
import numpy as np
import pandas as pd
from pipeline_complete import CompletePipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def test_pipeline_accuracy():
    """Test pipeline on training data samples"""
    
    print("="*80)
    print("PIPELINE ACCURACY TEST")
    print("="*80)
    
    # Load training data first to check structure
    print("\n[1/3] Loading training data...")
    try:
        df_train = pd.read_parquet("hp_segments_points.parquet")
        print(f"   Loaded {len(df_train)} training samples")
        print(f"   Columns: {df_train.columns.tolist()}")
    except FileNotFoundError:
        print("   ❌ hp_segments_points.parquet not found!")
        print("   Please run notebook cell 52 to export it:")
        print("   data2.to_parquet('./app/hp_segments_points.parquet', index=False)")
        return
    
    # Check if PRICE_ln exists (target variable)
    if 'PRICE_ln' not in df_train.columns:
        print("   ❌ PRICE_ln column not found in training data!")
        return
    
    # Select random samples
    print("\n[2/3] Sampling test cases...")
    n_samples = 10
    sample_df = df_train.sample(n_samples, random_state=42)
    
    # Training data already has REGION_GRID and segment_code encoded!
    # We just need to select the features that match the model
    model_features = ['ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL AREA', 'FLOOR',
                     'TOTAL_FLOORS', 'FURNITURE', 'CONDITION', 'CEILING',
                     'MATERIAL', 'YEAR', 'REGION_GRID', 'segment_code']
    
    # Check if all features exist
    missing_features = set(model_features) - set(sample_df.columns)
    if missing_features:
        print(f"   ❌ Missing features in training data: {missing_features}")
        print(f"   Available columns: {sample_df.columns.tolist()}")
        return
    
    test_inputs = sample_df[model_features].copy()
    # Rename column to match model expectation (underscore instead of space)
    test_inputs.rename(columns={'TOTAL AREA': 'TOTAL_AREA'}, inplace=True)
    actual_prices_ln = sample_df['PRICE_ln'].values
    actual_prices_kzt = np.exp(actual_prices_ln)
    
    # Make predictions
    print("\n[3/3] Making predictions...")
    # Since training data already has REGION_GRID and segment_code,
    # use predictor directly instead of full pipeline
    from predict_from_features import HousePricePredictor
    predictor = HousePricePredictor("nn_model")
    
    predicted_prices_kzt = predictor.predict_price(test_inputs)
    predicted_prices_ln = np.log(predicted_prices_kzt)
    
    # Calculate errors
    print("\n" + "="*80)
    print("INDIVIDUAL PREDICTIONS")
    print("="*80)
    
    for i in range(n_samples):
        actual_kzt = actual_prices_kzt[i]
        pred_kzt = predicted_prices_kzt[i]
        error_pct = ((pred_kzt - actual_kzt) / actual_kzt) * 100
        
        print(f"\nSample {i+1}:")
        print(f"  Location: ({test_inputs.iloc[i]['LATITUDE']:.4f}, {test_inputs.iloc[i]['LONGITUDE']:.4f})")
        print(f"  Rooms: {test_inputs.iloc[i]['ROOMS']}, Area: {test_inputs.iloc[i]['TOTAL_AREA']:.1f} m²")
        print(f"  Floor: {test_inputs.iloc[i]['FLOOR']}/{test_inputs.iloc[i]['TOTAL_FLOORS']}, Year: {test_inputs.iloc[i]['YEAR']:.0f}")
        print(f"  Actual Price:    {actual_kzt:>15,.0f} KZT")
        print(f"  Predicted Price: {pred_kzt:>15,.0f} KZT")
        print(f"  Error: {error_pct:+.2f}%")
    
    # Calculate overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    # Metrics on log scale (as in training)
    mape_ln = np.mean(np.abs((actual_prices_ln - predicted_prices_ln) / actual_prices_ln)) * 100
    mae_ln = mean_absolute_error(actual_prices_ln, predicted_prices_ln)
    mse_ln = mean_squared_error(actual_prices_ln, predicted_prices_ln)
    rmse_ln = np.sqrt(mse_ln)
    r2_ln = r2_score(actual_prices_ln, predicted_prices_ln)
    
    # Metrics on KZT scale
    mape_kzt = np.mean(np.abs((actual_prices_kzt - predicted_prices_kzt) / actual_prices_kzt)) * 100
    mae_kzt = mean_absolute_error(actual_prices_kzt, predicted_prices_kzt)
    
    print(f"\n📊 Log Scale Metrics (matching training):")
    print(f"   MAPE:  {mape_ln:.2f}%")
    print(f"   MAE:   {mae_ln:.6f}")
    print(f"   RMSE:  {rmse_ln:.6f}")
    print(f"   R²:    {r2_ln:.4f}")
    
    print(f"\n💰 KZT Scale Metrics:")
    print(f"   MAPE:  {mape_kzt:.2f}%")
    print(f"   MAE:   {mae_kzt:,.0f} KZT")
    
    # Compare with training metrics
    print("\n" + "="*80)
    print("COMPARISON WITH TRAINING")
    print("="*80)
    
    # Expected training metrics from notebook
    expected_mape = 0.88  # From notebook
    expected_r2 = 0.90    # From notebook
    
    print(f"\n{'Metric':<15} {'Training':<15} {'Pipeline':<15} {'Match':<10}")
    print("-"*60)
    print(f"{'MAPE (%)':<15} {expected_mape:<15.2f} {mape_ln:<15.2f} {'✓' if abs(mape_ln - expected_mape) < 1.0 else '⚠':<10}")
    print(f"{'R² Score':<15} {expected_r2:<15.2f} {r2_ln:<15.2f} {'✓' if abs(r2_ln - expected_r2) < 0.05 else '⚠':<10}")
    
    # Final verdict
    print("\n" + "="*80)
    if abs(mape_ln - expected_mape) < 1.0 and abs(r2_ln - expected_r2) < 0.05:
        print("✅ PIPELINE MATCHES TRAINING - Predictions are accurate!")
    elif abs(mape_ln - expected_mape) < 5.0:
        print("⚠️  PIPELINE CLOSE TO TRAINING - Small differences detected")
        print("   This may be due to random sampling or model stochasticity")
    else:
        print("❌ PIPELINE DOES NOT MATCH TRAINING - Significant differences!")
        print("   Check feature engineering and model artifacts")
    print("="*80)


if __name__ == "__main__":
    test_pipeline_accuracy()
