# FIXING PREDICTION MISMATCH - Step-by-Step Guide

## Problem
Web service predictions are 15-20% lower than training data prices because:
1. **REGION_GRID**: Web service calculates -1 (not found) instead of correct encoded values
2. **segment_code**: Web service assigns different codes than training data

## Root Cause
The model was trained with **integer-encoded** REGION_GRID values (0, 1, 2, ...) using LabelEncoder, but the web service was creating its own encoding that doesn't match training.

## Solution - Follow These Steps

### Step 1: Export Encoder Mappings from Notebook ✅

**Run these new cells in the notebook (cells 47-48):**

Cell 47 exports the LabelEncoder mapping for REGION_GRID:
```python
# This creates: app/region_grid_encoder.json
# Contains mapping like: {"Almaty": 0, "Astana": 1, ...}
```

Cell 48 analyzes segment_code to verify it's encoded correctly.

**After running, verify files exist:**
- ✅ `app/region_grid_encoder.json` - REGION_GRID name→code mapping
- ✅ `app/region_grid_lookup.json` - Grid cells→region names mapping

### Step 2: Code Updates ✅

**Already completed:**
- ✅ Updated `app/region_grid.py` to load region_grid_encoder.json
- ✅ Updated `app/predictor.py` to warn about -1 values
- ✅ Enhanced error detection

### Step 3: Test the Fix

**Option A: Test in Notebook**
```python
# Load web service components
import sys
sys.path.insert(0, './app')
from region_grid import get_region_grid_lookup

# Test with sample coordinates
lookup = get_region_grid_lookup()

# Test Almaty coordinates
test_coords = [(43.2220, 76.8512)]  # Almaty
region_codes = lookup.batch_lookup_region_codes(test_coords)
print(f"REGION_GRID for Almaty: {region_codes[0]}")
print(f"Should match training data value (not -1!)")
```

**Option B: Test in Web Service**
1. Start the web app: `streamlit run app.py`
2. Enter property details with known coordinates
3. Check prediction output - should show:
   - REGION_GRID = valid code (0, 1, 2, ...) NOT -1
   - segment_code = valid code NOT -1
4. Compare prediction with training data

### Step 4: Verify Training Data Compatibility

**Check sample from training data:**
```python
# In notebook
sample = data2.sample(1, random_state=42).iloc[0]
print(f"Sample property:")
print(f"  Coordinates: ({sample['LATITUDE']:.4f}, {sample['LONGITUDE']:.4f})")
print(f"  REGION_GRID: {sample['REGION_GRID']}")
print(f"  segment_code: {sample['segment_code']}")
print(f"  Actual price: {np.exp(sample['PRICE_ln']):,.0f} KZT")

# Test web service with SAME coordinates
# Prediction should be VERY close to actual price!
```

## Expected Results

### Before Fix:
- ❌ REGION_GRID = -1 (not found)
- ❌ segment_code = -1 or wrong value
- ❌ Predictions 15-20% lower than training data

### After Fix:
- ✅ REGION_GRID = valid code (0-N, matching training)
- ✅ segment_code = valid code (matching training)
- ✅ Predictions match training data (within 1-2% for same inputs)

## Troubleshooting

### Issue: "File not found: region_grid_encoder.json"
**Solution:** Run notebook cell 47 to export the encoder mapping

### Issue: REGION_GRID still showing -1
**Possible causes:**
1. Coordinates are outside Kazakhstan bounds
2. Coordinates are in area not covered by training data
3. region_grid_lookup.json is corrupted

**Debug:**
```python
# Check if coordinates are in grid
from app.region_grid import get_region_grid_lookup
lookup = get_region_grid_lookup()
print(f"Grid bounds: {lookup.bounds}")
print(f"Grid size: {lookup.grid_size}")
print(f"Total regions: {len(lookup.region_to_code)}")

# Test coordinate
region_name = lookup.lookup_region(43.2220, 76.8512)
region_code = lookup.lookup_region_code(43.2220, 76.8512)
print(f"Region name: {region_name}")
print(f"Region code: {region_code}")
```

### Issue: segment_code mismatch
**Solution:** Ensure segments_fine_heuristic_polygons.geojson exists in project root
- Web service uses spatial join with this file
- If missing, segment_code will be calculated differently

### Issue: Predictions still differ
**Check these:**
1. Verify all 13 features are present and correct
2. Check categorical encodings (FURNITURE: 1-3, CONDITION: 1-5, MATERIAL: 1-4)
3. Ensure feature order matches feature_list.json
4. Verify model artifacts (scaler_X, scaler_y) are from correct training run

## Testing Checklist

- [ ] Run notebook cell 47 to export region_grid_encoder.json
- [ ] Run notebook cell 48 to analyze segment_code
- [ ] Verify app/region_grid_encoder.json exists
- [ ] Restart web service to load new encoder
- [ ] Test prediction with known training data sample
- [ ] Verify REGION_GRID is not -1
- [ ] Verify segment_code is not -1
- [ ] Compare prediction with actual price (should be within 2-3%)
- [ ] Test with multiple properties across different regions

## Success Criteria

✅ Web service predictions match notebook predictions for same inputs
✅ REGION_GRID values are integers 0-N (not -1)
✅ segment_code values match training data
✅ Prediction error < 3% for properties in training data
✅ No warnings about -1 values in logs

## Next Steps After Fix

1. **Document the encoder files** in README
2. **Add validation** to reject properties with -1 codes
3. **Consider adding** region name display in UI (show which region was detected)
4. **Monitor predictions** and log any cases with -1 values
5. **Update model periodically** as new regions are added

---

## Technical Details

### Why This Matters

Machine learning models are sensitive to feature encoding. If the web service uses different encodings than training:

- **Training**: REGION_GRID="Almaty" → encoded as 5
- **Web Service**: REGION_GRID="Almaty" → encoded as 12
- **Result**: Model sees wrong value → wrong prediction

The fix ensures web service uses **EXACT SAME** encoding as training by:
1. Exporting encoder mapping from training (region_grid_encoder.json)
2. Loading this mapping in web service (region_grid.py)
3. Applying same encoding to new data

### File Dependencies

```
app/
├── region_grid_lookup.json       # Grid cells → region names
├── region_grid_encoder.json      # Region names → integer codes (NEW!)
├── region_grid.py               # Loads both files above
├── predictor.py                 # Uses region_grid.py
└── app.py                       # Calls predictor
```

All files must be present and from the same training run for predictions to match!
