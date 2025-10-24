"""Simple script to check REGION_GRID and segment_code distributions"""
import pandas as pd
import numpy as np

# Load training data
data = pd.read_parquet("app/hp_segments_points.parquet")

print("="*80)
print("TRAINING DATA FEATURE ANALYSIS")
print("="*80)

print(f"\nData shape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")

print("\n" + "="*80)
print("REGION_GRID Analysis:")
print("="*80)
print(f"Type: {data['REGION_GRID'].dtype}")
print(f"Unique values: {data['REGION_GRID'].nunique()}")
print(f"Value range: [{data['REGION_GRID'].min()}, {data['REGION_GRID'].max()}]")
print(f"\nTop 10 most common REGION_GRID values:")
print(data['REGION_GRID'].value_counts().head(10))

print("\n" + "="*80)
print("segment_code Analysis:")
print("="*80)
print(f"Type: {data['segment_code'].dtype}")
print(f"Unique values: {data['segment_code'].nunique()}")
print(f"Value range: [{data['segment_code'].min()}, {data['segment_code'].max()}]")
print(f"\nTop 10 most common segment_code values:")
print(data['segment_code'].value_counts().head(10))

print("\n" + "="*80)
print("Categorical Features Analysis:")
print("="*80)
print("\nFURNITURE values:")
print(data['FURNITURE'].value_counts().sort_index())
print("\nCONDITION values:")
print(data['CONDITION'].value_counts().sort_index())
print("\nMATERIAL values:")
print(data['MATERIAL'].value_counts().sort_index())

print("\n" + "="*80)
print("Sample rows:")
print("="*80)
print(data[['ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL_AREA', 'YEAR', 'REGION_GRID', 'segment_code', 'PRICE_ln']].head(3))

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("If REGION_GRID or segment_code in your web service inputs differ from")
print("the values above, predictions will be inaccurate!")
print("\nFor example:")
print(f"  - If training data has REGION_GRID range [0-{data['REGION_GRID'].max()}]")
print(f"    but web service calculates REGION_GRID = -1 (not found),")
print(f"    the model sees an out-of-distribution value → wrong prediction")
print("\nSOLUTION: Ensure web service uses THE SAME encoding/mapping as training data!")
print("="*80)
