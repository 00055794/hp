# Files to Remove Before GitHub Push

## Root Directory (102025_3/)

### Files to REMOVE:
1. **House_prices_KZ_data_preprocessed_wt_outliers_fromMLOps_Oct2025.pkl** - Large pickle file (not needed)
2. **hp_segments_points.parquet** - Large training data (duplicate, already in app/)
3. **segments_fine_heuristic_points.parquet** - Large file, not needed for deployment
4. **batch_test_template.csv** - Duplicate (already in app/)

### Documentation to KEEP (but could consolidate):
- APP_COMPARISON.md
- APP_NEW_COMPACT.md
- DEPLOYMENT.md
- FIX_PREDICTION_MISMATCH.md
- INTERFACE_UPDATE.md
- NEXT_STEPS.md
- QUICK_START.md
- TRAINING_INSTRUCTIONS.md
- UPDATE_SUMMARY.md

### Essential Files to KEEP:
- 3_GZ_House_Prices_Model_fromMLOps.ipynb (main notebook)
- region_grid_lookup.json (required for model)
- segments_fine_heuristic_polygons.geojson (required for model)
- segments_fine_heuristic_summary.csv (reference)
- Stat_KZ092025.xlsx (statistics data)
- regions_enriched.* (shapefiles - 5 files)
- stat.* (shapefiles - 5 files)
- render.yaml (deployment config)
- .gitignore, .dockerignore, .slugignore (config files)

## app/ Directory

### Files to REMOVE:
1. **__pycache__/** - Python cache (should be in .gitignore)
2. **.venv/** - Virtual environment (should be in .gitignore)
3. **debug_prediction_difference.py** - Debug script, not needed
4. **explain_log_mape.py** - Explanation script, not needed for deployment
5. **test_pipeline_accuracy.py** - Test script (optional, could keep for documentation)
6. **test_region_lookup.py** - Test script (optional)
7. **train_nn_correct.py** - Training script (could keep for reference)

### Files to KEEP:
- app.py (old interface, for reference)
- app_new.py (main application - **primary**)
- batch_test_template.csv (example template)
- feature_engineering.py (required)
- nn_inference.py (required)
- pipeline_complete.py (required)
- predictor.py (required)
- predict_from_features.py (required)
- region_grid.py (required)
- region_grid_encoder.json (required)
- region_grid_lookup.json (required)
- requirements.txt (required)
- nn_model/ (required - model artifacts)
- README.md, PIPELINE_README.md (documentation)

## data2/ Directory
- Check contents - likely contains intermediate files that can be removed

## kz_regions_shp/ Directory
- Keep if needed for visualization, otherwise remove

## Recommendation Summary

### MUST DELETE (Large/Unnecessary):
```
102025_3/House_prices_KZ_data_preprocessed_wt_outliers_fromMLOps_Oct2025.pkl
102025_3/hp_segments_points.parquet
102025_3/segments_fine_heuristic_points.parquet
102025_3/batch_test_template.csv (duplicate)
app/__pycache__/
app/.venv/
app/debug_prediction_difference.py
app/explain_log_mape.py
```

### OPTIONAL DELETE (Test/Debug):
```
app/test_pipeline_accuracy.py
app/test_region_lookup.py
app/train_nn_correct.py
```

### CHECK BEFORE DELETING:
```
data2/ (entire directory - check contents)
kz_regions_shp/ (if not used)
```

### .gitignore Should Include:
```
__pycache__/
*.pyc
.venv/
venv/
*.pkl
*.parquet (except specific ones)
.env
.DS_Store
```
