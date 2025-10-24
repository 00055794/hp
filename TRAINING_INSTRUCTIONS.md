# Training Instructions for House Price Model

## Summary

**All the features your web app uses ARE in the training data! ✅**

The columns in your notebook training data (`data2`) include:
- ✅ `ROOMS`
- ✅ `TOTAL_AREA` (note: has a space in notebook, will be normalized to `TOTAL_AREA`)
- ✅ `FLOOR`
- ✅ `TOTAL_FLOORS`
- ✅ `FURNITURE` (already encoded as 0/1/2)
- ✅ `CONDITION` (already encoded)
- ✅ `CEILING`
- ✅ `MATERIAL` (already encoded)
- ✅ `YEAR`
- ✅ `CITY_CAT`
- ✅ `srednmes_zarplata` (regional salary data)
- ✅ `segment_code` (encoded from `segment_id`)
- ✅ `PRICE_ln` (target - log price)

## Problem Identified

The previous `train_nn_updated.py` script was incorrectly trying to encode features like `BUILDING_TYPE`, `REGION`, `HOUSE_SIZE`, `YEAR_MONTH` which don't exist in your data. These were from a different section of your notebook.

The **correct** features are the ones your web app already collects!

## Solution

I created a corrected training script: `app/train_nn_correct.py`

This script:
1. Loads your data with the CORRECT features
2. Only encodes `segment_id` -> `segment_code` (using LabelEncoder)
3. Uses all other features as-is (they're already numeric/encoded)
4. Trains the PyTorch model matching your notebook exactly
5. Saves all artifacts for the web app

## How to Train the Model

### Step 1: Export Training Data from Notebook

First, run this cell in your notebook `3_GZ_House_Prices_Model_fromMLOps.ipynb` to export the training data:

```python
# Export hp_segments_points for training script
output_path = './data2/hp_segments_points.parquet'
os.makedirs('./data2', exist_ok=True)
hp_segments_points.to_parquet(output_path)
print(f"Exported training data to: {output_path}")
print(f"Shape: {hp_segments_points.shape}")
print(f"Columns: {list(hp_segments_points.columns)}")
```

### Step 2: Run the Training Script

Open a terminal in your project folder and run:

```powershell
cd c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3\app
python train_nn_correct.py --data ../data2/hp_segments_points.parquet --output ./nn_model
```

This will:
- Load your data
- Train for 100 epochs (with early stopping)
- Save the model to `app/nn_model/model.pt`
- Save scalers, feature list, and metadata

### Step 3: Test the Web App

After training, your web app should work without errors because the model will be trained on the exact same features the app collects.

## Training Parameters

You can customize training with these options:

```powershell
python train_nn_correct.py `
  --data ../data2/hp_segments_points.parquet `
  --output ./nn_model `
  --epochs 150 `
  --batch_size 64 `
  --lr 0.001 `
  --test_size 0.3 `
  --random_state 0
```

## Verifying the Fix

After training, check that `app/nn_model/feature_list.json` contains:

```json
[
  "ROOMS",
  "TOTAL_AREA",
  "FLOOR",
  "TOTAL_FLOORS",
  "FURNITURE",
  "CONDITION",
  "CEILING",
  "MATERIAL",
  "YEAR",
  "CITY_CAT",
  "srednmes_zarplata",
  "segment_code"
]
```

These match exactly what your web app form collects plus the regional features added by `predictor.py`.

## Next Steps

1. ✅ Export data from notebook (Step 1 above)
2. ✅ Run training script (Step 2 above)
3. ✅ Test predictions in web app
4. ✅ Commit and deploy to Render.com

## Questions?

If you get an error about missing columns, check:
1. Did you export the data after running the regional enrichment cell?
2. Does the exported data have all the columns listed above?
3. Run this in your notebook to verify:
   ```python
   print(hp_segments_points.columns.tolist())
   ```
