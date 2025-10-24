# Notebook-Matched Prediction Pipeline

## Overview
This directory contains a complete prediction pipeline that **exactly replicates** the notebook's training logic. The previous implementation had feature encoding mismatches causing predictions to be 15-20% lower than expected. This new implementation fixes all issues.

## Problem Solved
**Issue**: Web service predictions were 15-20% lower than training data predictions for identical inputs.

**Root Causes**:
1. **REGION_GRID encoding mismatch**: Web service was creating its own encoding instead of using the training LabelEncoder mapping
2. **segment_code assignment failures**: Points outside polygon boundaries got -1 instead of nearest segment

**Solution**: Complete rewrite using notebook-exact logic for all feature engineering and prediction steps.

## New Architecture

### Core Modules

#### 1. `feature_engineering.py`
Handles all feature preparation matching notebook logic:
- **REGION_GRID encoding**: Uses exported LabelEncoder mapping from `region_grid_encoder.json`
- **segment_code assignment**: Spatial join with nearest-neighbor fallback
- Validates all 11 base features + 2 computed features

#### 2. `predict_from_features.py`
Pure prediction module:
- Loads trained PyTorch model
- Applies MinMaxScaler transformations (exactly as in training)
- Returns predictions on log scale or KZT scale

#### 3. `pipeline_complete.py`
End-to-end pipeline combining feature engineering and prediction:
- Single prediction: `predict_single(input_dict)`
- Batch prediction: `predict_batch(df)`
- Returns prices in KZT and engineered features

#### 4. `app_new.py`
Streamlit web interface using the new pipeline:
- Single prediction with map interface
- Batch CSV upload
- Shows engineered features for transparency
- Clean, modern UI

#### 5. `test_pipeline_accuracy.py`
Validation script:
- Tests pipeline on training data samples
- Calculates MAPE, R², MAE, RMSE
- Compares with notebook metrics
- **Use this to verify accuracy before deployment**

## Required Files

### Model Artifacts (from notebook)
```
nn_model/
├── model.pt              # Trained PyTorch model
├── scaler_X.joblib       # Feature scaler (MinMaxScaler)
├── scaler_y.joblib       # Target scaler (MinMaxScaler)
├── feature_list.json     # List of 13 features in order
├── cat_mappings.json     # Categorical mappings
└── metadata.json         # Training metrics
```

### Feature Engineering Data
```
region_grid_lookup.json              # Grid cells -> region names
region_grid_encoder.json             # Region names -> integer codes (LabelEncoder)
../segments_fine_heuristic_polygons.geojson  # Segment polygons for spatial join
```

### Training Data (for testing)
```
hp_segments_points.parquet           # Full training data with PRICE_ln
```

## Usage

### 1. Test Pipeline Accuracy
```bash
cd app
python test_pipeline_accuracy.py
```

**Expected output**:
```
OVERALL METRICS
  MAPE:  0.88%
  R²:    0.90

✅ PIPELINE MATCHES TRAINING - Predictions are accurate!
```

### 2. Test Pipeline Standalone
```bash
python pipeline_complete.py
```

### 3. Run Web Service
```bash
streamlit run app_new.py
```

### 4. Use in Python
```python
from pipeline_complete import CompletePipeline

# Initialize once
pipeline = CompletePipeline()

# Single prediction
input_data = {
    'ROOMS': 2,
    'LONGITUDE': 76.9286,
    'LATITUDE': 43.2567,
    'TOTAL_AREA': 62.0,
    'FLOOR': 5,
    'TOTAL_FLOORS': 9,
    'FURNITURE': 2,
    'CONDITION': 3,
    'CEILING': 2.7,
    'MATERIAL': 2,
    'YEAR': 2015
}

price_kzt = pipeline.predict_single(input_data)
print(f"Predicted price: {price_kzt:,.0f} KZT")

# Batch prediction
import pandas as pd
df = pd.read_csv("properties.csv")
prices = pipeline.predict_batch(df)
```

## Feature Encoding Details

### Categorical Features (1-based)
- **FURNITURE**: 1=No, 2=Partial, 3=Full
- **CONDITION**: 1=Poor, 2=Fair, 3=Good, 4=Excellent, 5=Perfect
- **MATERIAL**: 1=Brick, 2=Panel, 3=Monolith, 4=Other

### Auto-Computed Features

#### REGION_GRID
1. **Grid Snapping**: Snap (lat, lon) to 0.01° grid cell
2. **Region Lookup**: Look up region name from `region_grid_lookup.json`
3. **Encoding**: Encode region name using `region_grid_encoder.json` (LabelEncoder mapping)
4. **Result**: Integer 0-35 (36 regions)

**Example**:
```
(43.2567, 76.9286) 
  -> Grid: (43.26, 76.93)
  -> Region: "Almaty_r-n"
  -> Code: 7
```

#### segment_code
1. **Spatial Join**: Create Point geometry and join with segment polygons
2. **Fallback**: If no polygon contains point, find nearest polygon
3. **Encoding**: Encode segment_id using sorted unique values (LabelEncoder equivalent)
4. **Result**: Integer 0-N (N = number of segments)

## Validation Checklist

Before deployment, verify:

- [ ] `test_pipeline_accuracy.py` shows MAPE < 1.5% and R² > 0.88
- [ ] `region_grid_encoder.json` contains 36 regions
- [ ] `segments_fine_heuristic_polygons.geojson` exists and has valid geometries
- [ ] No warnings about REGION_GRID=-1 or segment_code=-1 in test output
- [ ] Model artifacts are from latest training run
- [ ] All dependencies installed (`pip install -r requirements.txt`)

## Troubleshooting

### "REGION_GRID=-1" warnings
- Check that `region_grid_encoder.json` exists and matches training
- Verify coordinates are within Kazakhstan (40-55°N, 46-87°E)
- Check grid_size parameter (default 0.01)

### "segment_code=-1" warnings
- Verify `segments_fine_heuristic_polygons.geojson` exists
- Check CRS is EPSG:4326
- Ensure geometries are valid (no null/empty polygons)
- Should be rare with nearest-neighbor fallback

### Predictions still don't match
1. Run `test_pipeline_accuracy.py` to see exact metrics
2. Check model artifacts are from correct training run
3. Verify feature order matches `feature_list.json`
4. Compare scaler parameters with notebook

## Deployment Notes

### Environment Variables
None required - all paths are relative to app directory.

### Performance
- First load: ~5-10 seconds (loads model, segments, grids)
- Single prediction: ~50-100ms
- Batch (1000 properties): ~5-10 seconds

### Memory
- Base: ~200MB (model + artifacts)
- Per request: ~1-2MB
- Segments GeoJSON: ~50MB

## Changes from Old Implementation

| Component | Old | New |
|-----------|-----|-----|
| Feature Engineering | predictor.py | feature_engineering.py |
| Prediction | nn_inference.py | predict_from_features.py |
| Pipeline | Manual in app.py | pipeline_complete.py |
| REGION_GRID | region_grid.py (broken) | feature_engineering.py (fixed) |
| segment_code | predictor.py (no fallback) | feature_engineering.py (with fallback) |
| Web App | app.py (complex) | app_new.py (simple) |
| Testing | check_training_data.py | test_pipeline_accuracy.py |

## Migration Path

1. **Test new pipeline**: `python test_pipeline_accuracy.py`
2. **Verify accuracy**: Check MAPE and R² match training
3. **Test web app**: `streamlit run app_new.py`
4. **Compare predictions**: Make same prediction in old and new app
5. **Switch**: Rename `app_new.py` to `app.py`
6. **Archive old files**: Move old predictor.py, nn_inference.py to `old/` folder

## Support

If predictions still don't match training:
1. Check that `region_grid_encoder.json` was exported from notebook (cell 47)
2. Verify training data was saved with: `data2.to_parquet('./app/hp_segments_points.parquet')`
3. Ensure model artifacts are from the same training run
4. Run notebook diagnostic cells (58-59) to compare predictions directly

## Success Criteria

✅ Pipeline is correct when:
- `test_pipeline_accuracy.py` shows MAPE < 1.5%
- R² Score > 0.88
- No systematic bias (errors distributed around 0%)
- Individual predictions within ±5% of training data predictions

---
**Last Updated**: 2025-10-24
**Notebook Version**: 3_GZ_House_Prices_Model_fromMLOps.ipynb (cells 1-60 executed)
**Training Data**: 588,324 properties, R²=0.90, MAPE=0.88%
