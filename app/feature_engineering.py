"""
Feature engineering pipeline matching the notebook's data2 preparation.

This module prepares raw input data to match the exact features used in training:
- ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR, TOTAL_FLOORS
- FURNITURE, CONDITION, CEILING, MATERIAL, YEAR
- REGION_GRID (encoded), segment_code (encoded)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import Point


class FeatureEngineer:
    """
    Prepares features to match training data exactly.
    """
    
    def __init__(self, 
                 region_grid_lookup_path="region_grid_lookup.json",
                 region_grid_encoder_path="region_grid_encoder.json",
                 segments_geojson_path="../segments_fine_heuristic_polygons.geojson"):
        """
        Initialize feature engineering with required lookup files.
        
        Args:
            region_grid_lookup_path: Path to region grid lookup JSON (lat/lon -> region name)
            region_grid_encoder_path: Path to region encoder JSON (region name -> integer code)
            segments_geojson_path: Path to segments polygons GeoJSON
        """
        # Load region grid lookup
        with open(region_grid_lookup_path, 'r', encoding='utf-8') as f:
            region_grid_data = json.load(f)
        # Extract the 'grid' dictionary from the JSON structure
        self.region_grid_lookup = region_grid_data.get('grid', {})
        print(f"✓ Loaded region grid lookup: {len(self.region_grid_lookup)} cells")
        
        # Load region encoder mapping
        with open(region_grid_encoder_path, 'r', encoding='utf-8') as f:
            self.region_encoder = json.load(f)
        print(f"✓ Loaded region encoder: {len(self.region_encoder)} regions")
        
        # Load segments polygons
        self.segments_gdf = gpd.read_file(segments_geojson_path)
        print(f"✓ Loaded segments: {len(self.segments_gdf)} polygons")
        
        # Build segment_id encoder (LabelEncoder equivalent)
        # This must match the notebook's LabelEncoder order (sorted unique values)
        segment_ids = sorted(self.segments_gdf['segment_id'].dropna().unique())
        self.segment_encoder = {seg_id: code for code, seg_id in enumerate(segment_ids)}
        print(f"✓ Created segment encoder: {len(self.segment_encoder)} segments")
    
    def lookup_region_code(self, lat, lon, grid_size=0.01):
        """
        Look up REGION_GRID code for a coordinate.
        
        Matches the notebook's batch_lookup_regions + LabelEncoder pipeline:
        1. Snap lat/lon to grid cell (integer grid coordinates)
        2. Look up region name from grid
        3. Encode region name to integer using LabelEncoder mapping
        
        Args:
            lat: Latitude
            lon: Longitude
            grid_size: Grid cell size (default 0.01 degrees)
            
        Returns:
            Integer region code (0-35) or -1 if not found
        """
        # Snap to grid using INTEGER coordinates (same as notebook)
        # This matches the notebook logic: round(lat / grid_size)
        lat_grid = int(round(lat / grid_size))
        lon_grid = int(round(lon / grid_size))
        grid_key = f"{lat_grid},{lon_grid}"
        
        # Look up region name
        region_name = self.region_grid_lookup.get(grid_key)
        
        if region_name is None or region_name == "Unknown":
            return -1
        
        # Encode region name to integer
        return self.region_encoder.get(region_name, -1)
    
    def lookup_segment_code(self, lat, lon):
        """
        Look up segment_code for a coordinate using spatial join.
        
        Matches the notebook's segment assignment:
        1. Create point geometry
        2. Spatial join with segment polygons
        3. Encode segment_id to integer
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Integer segment code or -1 if not found
        """
        # Create point
        point = Point(lon, lat)
        point_gdf = gpd.GeoDataFrame([{'geometry': point}], crs='EPSG:4326')
        
        # Spatial join
        joined = gpd.sjoin(point_gdf, self.segments_gdf[['segment_id', 'geometry']], 
                          how='left', predicate='within')
        
        if len(joined) == 0 or pd.isna(joined['segment_id'].iloc[0]):
            # Fallback: Find nearest segment
            distances = self.segments_gdf.geometry.distance(point)
            nearest_idx = distances.idxmin()
            segment_id = self.segments_gdf.loc[nearest_idx, 'segment_id']
        else:
            segment_id = joined['segment_id'].iloc[0]
        
        # Encode segment_id to integer
        return self.segment_encoder.get(segment_id, -1)
    
    def prepare_features(self, input_data):
        """
        Prepare features from raw input data.
        
        Args:
            input_data: dict or DataFrame with raw feature values
                Required keys: ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR, 
                               TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, 
                               MATERIAL, YEAR
        
        Returns:
            pandas DataFrame with all 13 features in correct order
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Validate required base features
        required_base = ['ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL_AREA', 'FLOOR',
                        'TOTAL_FLOORS', 'FURNITURE', 'CONDITION', 'CEILING',
                        'MATERIAL', 'YEAR']
        
        missing = set(required_base) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Add REGION_GRID (encoded)
        df['REGION_GRID'] = df.apply(
            lambda row: self.lookup_region_code(row['LATITUDE'], row['LONGITUDE']),
            axis=1
        )
        
        # Add segment_code (encoded)
        df['segment_code'] = df.apply(
            lambda row: self.lookup_segment_code(row['LATITUDE'], row['LONGITUDE']),
            axis=1
        )
        
        # Warn if encoding failed
        if (df['REGION_GRID'] == -1).any():
            n_missing = (df['REGION_GRID'] == -1).sum()
            print(f"⚠️  WARNING: {n_missing} rows have REGION_GRID=-1 (not found in grid)")
        
        if (df['segment_code'] == -1).any():
            n_missing = (df['segment_code'] == -1).sum()
            print(f"⚠️  WARNING: {n_missing} rows have segment_code=-1 (no segment assigned)")
        
        # Feature order must match training (from feature_list.json)
        feature_order = ['ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL_AREA', 'FLOOR',
                        'TOTAL_FLOORS', 'FURNITURE', 'CONDITION', 'CEILING',
                        'MATERIAL', 'YEAR', 'REGION_GRID', 'segment_code']
        
        return df[feature_order]


def main():
    """Example usage"""
    print("="*80)
    print("Feature Engineering Pipeline Test")
    print("="*80)
    
    # Initialize
    engineer = FeatureEngineer()
    
    # Test with example data (Almaty)
    test_input = {
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
    
    print("\nInput data:")
    for k, v in test_input.items():
        print(f"  {k}: {v}")
    
    # Prepare features
    features_df = engineer.prepare_features(test_input)
    
    print("\nEngineered features:")
    print(features_df.to_string())
    
    print("\n✓ Feature engineering test completed")


if __name__ == "__main__":
    main()
