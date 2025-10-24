"""
Complete prediction pipeline matching notebook exactly.

This script combines feature engineering and model prediction to produce
price estimates that match the training notebook's logic.
"""
import numpy as np
import pandas as pd
from feature_engineering import FeatureEngineer
from predict_from_features import HousePricePredictor


class CompletePipeline:
    """End-to-end pipeline: raw input -> features -> prediction"""
    
    def __init__(self, 
                 model_dir="nn_model",
                 region_grid_lookup="region_grid_lookup.json",
                 region_grid_encoder="region_grid_encoder.json",
                 segments_geojson="../segments_fine_heuristic_polygons.geojson"):
        """
        Initialize complete pipeline.
        
        Args:
            model_dir: Directory with trained model artifacts
            region_grid_lookup: Path to region grid lookup JSON
            region_grid_encoder: Path to region encoder JSON  
            segments_geojson: Path to segments GeoJSON
        """
        print("="*80)
        print("Initializing Complete Prediction Pipeline")
        print("="*80)
        
        # Initialize feature engineer
        print("\n[1/2] Loading feature engineering components...")
        self.engineer = FeatureEngineer(
            region_grid_lookup_path=region_grid_lookup,
            region_grid_encoder_path=region_grid_encoder,
            segments_geojson_path=segments_geojson
        )
        
        # Initialize predictor
        print("\n[2/2] Loading trained model...")
        self.predictor = HousePricePredictor(model_dir=model_dir)
        
        print("\n" + "="*80)
        print("✅ Pipeline ready for predictions")
        print("="*80)
    
    def predict_single(self, input_dict, return_features=False):
        """
        Predict price for a single property.
        
        Args:
            input_dict: Dictionary with property features
                Required: ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR,
                         TOTAL_FLOORS, FURNITURE, CONDITION, CEILING,
                         MATERIAL, YEAR
            return_features: If True, also return engineered features
        
        Returns:
            If return_features=False: price in KZT
            If return_features=True: (price_kzt, features_df)
        """
        # Engineer features
        features_df = self.engineer.prepare_features(input_dict)
        
        # Predict
        price_ln = self.predictor.predict(features_df)[0]
        price_kzt = np.exp(price_ln)
        
        if return_features:
            return price_kzt, features_df
        else:
            return price_kzt
    
    def predict_batch(self, df):
        """
        Predict prices for multiple properties.
        
        Args:
            df: DataFrame with property features (same columns as predict_single)
        
        Returns:
            numpy array of prices in KZT
        """
        # Engineer features
        features_df = self.engineer.prepare_features(df)
        
        # Predict
        prices_kzt = self.predictor.predict_price(features_df)
        
        return prices_kzt


def main():
    """Test pipeline with example predictions"""
    print("\n")
    print("="*80)
    print("TESTING COMPLETE PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Test cases
    test_cases = [
        {
            'name': 'Almaty - 2-room apartment',
            'data': {
                'ROOMS': 2,
                'LONGITUDE': 76.9286,
                'LATITUDE': 43.2567,
                'TOTAL_AREA': 62.0,
                'FLOOR': 5,
                'TOTAL_FLOORS': 9,
                'FURNITURE': 2,  # Partial
                'CONDITION': 3,  # Good
                'CEILING': 2.7,
                'MATERIAL': 2,   # Panel
                'YEAR': 2015
            }
        },
        {
            'name': 'Astana - 3-room apartment',
            'data': {
                'ROOMS': 3,
                'LONGITUDE': 71.4704,
                'LATITUDE': 51.1694,
                'TOTAL_AREA': 85.0,
                'FLOOR': 7,
                'TOTAL_FLOORS': 12,
                'FURNITURE': 3,  # Full
                'CONDITION': 4,  # Excellent
                'CEILING': 2.8,
                'MATERIAL': 3,   # Monolith
                'YEAR': 2018
            }
        }
    ]
    
    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 80)
        
        # Show input
        print("Input features:")
        for k, v in test['data'].items():
            print(f"  {k:15s}: {v}")
        
        # Make prediction
        price, features = pipeline.predict_single(test['data'], return_features=True)
        
        # Show engineered features
        print("\nEngineered features:")
        print(f"  REGION_GRID     : {features['REGION_GRID'].iloc[0]}")
        print(f"  segment_code    : {features['segment_code'].iloc[0]}")
        
        # Show prediction
        print(f"\n💰 Predicted Price: {price:,.0f} KZT")
        print(f"   (approx ${price/460:,.0f} USD at 460 KZT/USD)")
    
    print("\n" + "="*80)
    print("✅ Pipeline test completed successfully")
    print("="*80)


if __name__ == "__main__":
    main()
