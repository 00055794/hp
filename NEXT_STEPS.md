# NEXT STEPS - Quick Reference Guide

## 🚨 IMMEDIATE ACTION REQUIRED

### Step 1: Export Training Data from Notebook
Open `3_GZ_House_Prices_Model_fromMLOps.ipynb` and run:

**After Cell 46** (where REGION_GRID is encoded and data2 is prepared):

```python
# Export the final training data with all 13 features
data2.to_parquet('hp_segments_points.parquet', index=False)
print(f"Exported {len(data2)} rows with columns: {list(data2.columns)}")

# Verify the exported data has correct features
print("\nExpected features for training:")
expected = ['ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL_AREA', 'FLOOR', 'TOTAL_FLOORS',
            'FURNITURE', 'CONDITION', 'CEILING', 'MATERIAL', 'YEAR', 
            'REGION_GRID', 'segment_code', 'PRICE_ln']
print(expected)
print("\nActual columns:")
print(list(data2.columns))
print("\nMissing:", [c for c in expected if c not in data2.columns])
print("Extra:", [c for c in data2.columns if c not in expected])
```

Then copy the file:
```powershell
Copy-Item "c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3\hp_segments_points.parquet" -Destination "c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3\app\hp_segments_points.parquet"
```

### Step 2: Retrain the Model
```powershell
cd c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3\app
python train_nn_correct.py --data hp_segments_points.parquet --output nn_model
```

**Expected output:**
```
Loading data from: hp_segments_points.parquet
Loaded XXXX rows, XX columns
...
Features shape: (XXXX, 13)
Feature columns: ['ROOMS', 'LONGITUDE', 'LATITUDE', 'TOTAL_AREA', 'FLOOR', 'TOTAL_FLOORS', 'FURNITURE', 'CONDITION', 'CEILING', 'MATERIAL', 'YEAR', 'REGION_GRID', 'segment_code']
...
Epoch 100/100: Train Loss=X.XXXX, Val Loss=X.XXXX
...
Saved model artifacts to: nn_model/
  - model.pt
  - scaler_X.joblib
  - scaler_y.joblib
  - feature_list.json
  - metadata.json
```

### Step 3: Verify Model Artifacts
```powershell
# Check feature_list.json has 13 features
Get-Content app\nn_model\feature_list.json

# Check metadata
Get-Content app\nn_model\metadata.json
```

**Expected feature_list.json:**
```json
["ROOMS", "LONGITUDE", "LATITUDE", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION", "CEILING", "MATERIAL", "YEAR", "REGION_GRID", "segment_code"]
```

**Expected metadata.json:**
```json
{
  "r2": 0.90,
  "mape": 0.87,
  "input_dim": 13,
  ...
}
```

### Step 4: Test the Web App
```powershell
cd c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3\app
streamlit run app.py
```

**Test predictions with:**
1. **Almaty apartment:**
   - Rooms: 2
   - Area: 60 m²
   - Year: 2015
   - Floor: 5
   - Total Floors: 9
   - Ceiling: 2.7 m
   - Material: Brick
   - Furniture: No
   - Condition: Good
   - Latitude: 43.2220
   - Longitude: 76.8512

2. **Astana apartment:**
   - Rooms: 3
   - Area: 75 m²
   - Year: 2018
   - Floor: 7
   - Total Floors: 12
   - Ceiling: 2.8 m
   - Material: Monolithic
   - Furniture: Partial
   - Condition: Excellent
   - Latitude: 51.1694
   - Longitude: 71.4491

**Verify:**
- ✅ No errors during prediction
- ✅ REGION_GRID appears in enriched data (should be integer, not -1)
- ✅ segment_code is assigned (should be integer, not -1)
- ✅ Prediction is reasonable (e.g., 20-50 million KZT for typical apartment)

### Step 5: Compare Predictions (Validation)
**In notebook:**
```python
# Use the same inputs as web app test
test_input = pd.DataFrame({
    'ROOMS': [2],
    'LONGITUDE': [76.8512],
    'LATITUDE': [43.2220],
    'TOTAL_AREA': [60.0],
    'FLOOR': [5],
    'TOTAL_FLOORS': [9],
    'FURNITURE': [1],  # No=1
    'CONDITION': [2],  # Good=2
    'CEILING': [2.7],
    'MATERIAL': [2],  # Brick=2
    'YEAR': [2015],
    'REGION_GRID': [lookup_region_code(43.2220, 76.8512)],  # Calculate using your function
    'segment_code': [X]  # Use appropriate segment_code
})

# Predict using notebook model
X_test = scaler_X.transform(test_input)
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32))
y_pred_ln = scaler_y.inverse_transform(y_pred_scaled.detach().numpy())
y_pred_kzt = np.exp(y_pred_ln)
print(f"Notebook prediction: {y_pred_kzt[0]:,.0f} KZT")
```

**Compare with web app prediction - should match within ±1%**

## 📋 Checklist

- [ ] Run notebook Cell 46 to prepare data2 with REGION_GRID
- [ ] Export hp_segments_points.parquet from notebook
- [ ] Copy parquet file to app/ folder
- [ ] Run train_nn_correct.py to retrain model
- [ ] Verify feature_list.json has 13 features
- [ ] Verify metadata.json shows input_dim=13
- [ ] Test web app with sample predictions
- [ ] Compare notebook vs web app predictions (should match)
- [ ] Test batch CSV upload with multiple properties
- [ ] Update TRAINING_INSTRUCTIONS.md if needed

## ⚠️ Common Issues

**Issue: "File not found: region_grid_lookup.json"**
- **Solution:** Already copied to app/, but verify it exists

**Issue: "Expected 12 features, got 13" or vice versa**
- **Solution:** Old model still loaded. Delete old nn_model/ contents and retrain

**Issue: "REGION_GRID not found in columns"**
- **Solution:** Training data doesn't have REGION_GRID. Re-export from notebook after Cell 46

**Issue: "All REGION_GRID values are -1"**
- **Solution:** region_grid_lookup.json is empty or coordinates are outside bounds

**Issue: Predictions way off from notebook**
- **Solution:** Check feature order in feature_list.json matches training data columns
- Verify categorical encodings (FURNITURE, CONDITION, MATERIAL) are 1-based

## 📊 Expected Results

After retraining with new 13-feature model:
- **R² Score:** ~0.90 (same or better than old model)
- **MAPE:** <1% (very accurate)
- **Training Time:** 2-5 minutes on CPU (100 epochs)
- **Model Size:** ~50KB (model.pt)
- **Features:** 13 (was 12)
- **Prediction Speed:** <100ms per property

## 🎯 Success Criteria

✅ Model trains without errors
✅ feature_list.json has exactly 13 features
✅ Web app makes predictions without errors
✅ REGION_GRID values are integers (not -1)
✅ Predictions match notebook predictions
✅ Performance metrics (R², MAPE) are similar to old model

---

**Date:** January 2025
**Status:** Code ready, awaiting model retraining
**Author:** GitHub Copilot
