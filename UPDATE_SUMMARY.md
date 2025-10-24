# Web Service Update Summary - NEW 13-Feature Model

## Overview
Updated the entire web service to match the new 13-feature model from the notebook `3_GZ_House_Prices_Model_fromMLOps.ipynb`.

## Changes Made

### 1. New Feature Set
**OLD Model (12 features):**
- ROOMS, TOTAL_AREA, FLOOR, TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, MATERIAL, YEAR, CITY_CAT, srednmes_zarplata, segment_code

**NEW Model (13 features):**
- ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR, TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, MATERIAL, YEAR, REGION_GRID, segment_code

**Key Changes:**
- ✅ Added LONGITUDE, LATITUDE as direct model inputs
- ✅ Added REGION_GRID (int32) calculated from coordinates via grid lookup
- ✅ Removed CITY_CAT (replaced by REGION_GRID)
- ✅ Removed srednmes_zarplata (salary data no longer used)

### 2. Files Updated

#### `app/region_grid.py` (NEW FILE)
- Created `RegionGridLookup` class to handle region grid calculations
- Implements grid-based coordinate-to-region mapping (0.01° cells ≈ 1km squares)
- Methods:
  - `lookup_region(lat, lon)` - Get region name from coordinates
  - `lookup_region_code(lat, lon)` - Get integer-encoded region code
  - `batch_lookup_region_codes(coords)` - Batch processing for DataFrames
- Uses `region_grid_lookup.json` (copied from notebook output)

#### `app/predictor.py`
- Updated `attach_region_features()` function:
  - Now calculates REGION_GRID using RegionGridLookup
  - Keeps segment_code calculation for market segmentation
  - Removed srednmes_zarplata and chislennost_naseleniya_092025
  - Returns DataFrame with all 13 required features
  - Ensures int32 dtype for REGION_GRID and segment_code

#### `app/train_nn_correct.py`
- Updated docstring to reflect NEW 13-feature model
- Changed expected features list from 12 to 13
- Updated `load_and_preprocess_data()`:
  - NO LONGER drops LONGITUDE, LATITUDE (now part of model)
  - Drops CITY_CAT and srednmes_zarplata (old features)
  - Expects REGION_GRID in input data
- Feature order: ROOMS, LONGITUDE, LATITUDE, TOTAL_AREA, FLOOR, TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, MATERIAL, YEAR, REGION_GRID, segment_code

#### `app/app.py`
- Updated `predict_on_df()` to remove references to old columns
- Changed enrichment from 4 columns (segment_code, region_name, srednmes_zarplata, chislennost) to 2 (segment_code, REGION_GRID)
- Form inputs remain unchanged (already correct - 11 user inputs)

#### `app/region_grid_lookup.json`
- Copied from main folder to app folder
- Contains grid-based region mapping data from notebook Cell 11

### 3. Model Architecture (Unchanged)
- PyTorch neural network: 64→16→1
- Input dimension: **13** (was 12)
- MinMaxScaler for both features and target
- Log-transformed price target (PRICE_ln)

## Next Steps - CRITICAL

### 1. Export Updated Training Data from Notebook
You need to run the notebook and export the data with REGION_GRID:

```python
# In notebook, after Cell 46 (where REGION_GRID is encoded):
data2.to_parquet('hp_segments_points.parquet', index=False)
```

Copy this file to the `app/` folder (or parent 102025_3/ folder).

### 2. Retrain the Model
Run the updated training script:

```powershell
cd app
python train_nn_correct.py --data ../hp_segments_points.parquet --output nn_model
```

This will generate NEW model artifacts:
- `nn_model/model.pt` - Neural network weights (input_dim=13)
- `nn_model/scaler_X.joblib` - Feature scaler for 13 features
- `nn_model/scaler_y.joblib` - Target scaler
- `nn_model/feature_list.json` - List of 13 features in correct order
- `nn_model/metadata.json` - Training metrics (R², MAPE, etc.)
- `nn_model/cat_mappings.json` - Categorical mappings (if any)

### 3. Verify Feature Order
After retraining, check `nn_model/feature_list.json` should contain:
```json
[
  "ROOMS",
  "LONGITUDE", 
  "LATITUDE",
  "TOTAL_AREA",
  "FLOOR",
  "TOTAL_FLOORS",
  "FURNITURE",
  "CONDITION",
  "CEILING",
  "MATERIAL",
  "YEAR",
  "REGION_GRID",
  "segment_code"
]
```

### 4. Test the Web App
```powershell
cd app
streamlit run app.py
```

Test with known coordinates:
- Almaty: ~43.2220, 76.8512
- Astana: ~51.1694, 71.4491
- Shymkent: ~42.3417, 69.5901

Verify:
- REGION_GRID is calculated correctly (should be integer like 1, 2, 3...)
- segment_code is assigned
- Predictions are reasonable

### 5. End-to-End Validation
Compare predictions:
1. Make prediction in notebook with same inputs
2. Make prediction in web app with same inputs
3. Results should be identical (within floating point precision)

## File Checklist

### Files That MUST Be Present in app/ Folder:
- ✅ `region_grid.py` - Region grid lookup module
- ✅ `region_grid_lookup.json` - Grid mapping data (copied from notebook)
- ⏳ `hp_segments_points.parquet` - Training data with 13 features (NEEDS EXPORT)
- ⏳ `nn_model/model.pt` - Retrained model (NEEDS RETRAINING)
- ⏳ `nn_model/scaler_X.joblib` - New scaler (NEEDS RETRAINING)
- ⏳ `nn_model/scaler_y.joblib` - New scaler (NEEDS RETRAINING)
- ⏳ `nn_model/feature_list.json` - 13 features (NEEDS RETRAINING)

### Files Already Updated:
- ✅ `predictor.py` - Uses region_grid module
- ✅ `train_nn_correct.py` - Expects 13 features
- ✅ `app.py` - Simplified enrichment logic
- ✅ `nn_inference.py` - No changes needed (uses feature_list.json dynamically)

## Expected Performance
- Model should maintain similar performance to old model
- R² ≈ 0.90
- MAPE < 1%
- REGION_GRID provides spatial context without needing salary data
- Simpler feature engineering pipeline

## Troubleshooting

### If predictions fail:
1. Check `region_grid_lookup.json` exists in app folder
2. Verify model was retrained with 13 features (check metadata.json)
3. Check feature_list.json has exactly 13 features in correct order
4. Verify REGION_GRID and segment_code are int32 dtype

### If REGION_GRID is always -1:
- region_grid_lookup.json is missing or empty
- Coordinates are outside Kazakhstan bounds
- RegionGridLookup initialization failed

### If error "expected 12 features, got 13":
- Old model artifacts still in nn_model/ folder
- Need to retrain with new training script

## User Input Flow
1. User enters 11 values: ROOMS, TOTAL_AREA, FLOOR, TOTAL_FLOORS, FURNITURE, CONDITION, CEILING, MATERIAL, YEAR, LATITUDE, LONGITUDE
2. System calculates: REGION_GRID (from region_grid_lookup.json), segment_code (from spatial join)
3. Total 13 features passed to model
4. Model predicts log price, inverse transform, return KZT

## Date: 2025
## Status: ✅ Code Updated, ⏳ Model Retraining Required
