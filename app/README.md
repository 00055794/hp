# Kazakhstan House Price Predictor (KZT)

A free, fast, open-source Streamlit web app to predict apartment prices in Kazakhstan in real KZT, not ln-scaled. Supports:

- Single prediction via a form
- Batch predictions via CSV/XLSX upload
- Interactive Folium map with clickable markers showing predicted price and details

The app enriches inputs with regional socio-economic features (`srednmes_zarplata`, `chislennost_naseleniya_092025`) by snapping each point to the nearest centroid from `Stat_KZ092025.xlsx`.

## Files

- `app.py` — Streamlit UI
- `predictor.py` — Inference helpers (region stats lookup, validation, prediction)
- `train_model.py` — Minimal training script producing `model.joblib` (predicts ln(price))
- `train_nn.py` — PyTorch neural network trainer that exports NN artifacts for the app
- `requirements.txt` — Python dependencies
- `template.csv` — Example batch upload format

Place `Stat_KZ092025.xlsx` in this same folder (copy from `102025/Stat_KZ092025.xlsx`). If your stats file is elsewhere, set env var `STATS_XLSX`.

## Setup (Windows PowerShell)

1) Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Train a quick baseline model (uses your preprocessed dataset with `PRICE_ln`):

```powershell
# Use the data under 102025
python train_model.py --data ..\House_prices_KZ_data_preprocessed_wt_outliers_fromMLOps.pkl --with-stats --out model.joblib
```

Notes:
- The script expects `PRICE_ln` in the dataset. If only `PRICE` exists, it will create `PRICE_ln = log(PRICE)`.
- It also uses `srednmes_zarplata` and `chislennost_naseleniya_092025` if present.

4) Copy stats file next to the app (if not already):

```powershell
Copy-Item ..\Stat_KZ092025.xlsx .\Stat_KZ092025.xlsx
```

5) Run the app:

```powershell
$env:MODEL_PATH="$(Get-Location)\model.joblib"; $env:STATS_XLSX="$(Get-Location)\Stat_KZ092025.xlsx"; streamlit run app.py
```

The app opens in your browser (default: http://localhost:8501).

### Neural Network option

If you prefer to use the neural network (as in your notebooks):

```powershell
python train_nn.py --data ..\House_prices_KZ_data_preprocessed_wt_outliers_fromMLOps.pkl --outdir .\nn_model
```

This will create `nn_model/` containing:

- `model.pt` — trained PyTorch model weights
- `scaler_X.joblib` — feature MinMax scaler
- `scaler_y.joblib` — ln(target) MinMax scaler
- `feature_list.json` — exact feature order used in training
- `cat_mappings.json` — integer encodings for FURNITURE, CONDITION, MATERIAL

To run the app preferring the NN model, ensure `nn_model/` sits next to `app.py` or set:

```powershell
$env:NN_MODEL_DIR = "$(Get-Location)\nn_model"; streamlit run app.py
```

If NN artifacts are present, the app will use them and return predictions directly in real KZT (by inverse scaling and exponentiating the ln-target). Otherwise, it falls back to the baseline scikit-learn pipeline.

## Input schema

Required columns for predictions (case sensitive):

- ROOMS (int)
- TOTAL_AREA (float) — can also upload as `TOTAL AREA` and it will be normalized
- FLOOR (int)
- TOTAL_FLOORS (int)
- FURNITURE (str): e.g., No | Partial | Full
- CONDITION (str): Needs renovation | Good | Excellent
- CEILING (float): height in meters
- MATERIAL (str): Panel | Brick | Monolithic | Mixed
- YEAR (int)
- LATITUDE (float)
- LONGITUDE (float)

You can start from `template.csv`.

## How predictions are converted to KZT

The model is trained to predict ln(price). At inference the app computes: `price_kzt = exp(predicted_ln)`. Values are displayed with thousands separators.

## Map visualization

- Centered on Kazakhstan by default
- Markers show predicted KZT price in tooltip; click to see full details

Note on segmentation and enrichment:

- By default, the app attaches `segment_code`, `region_name`, `srednmes_zarplata`, and `chislennost_naseleniya_092025` by nearest regional centroid from the stats Excel.
- Advanced (optional): build precise polygons and stable segment_code mapping using the provided CLI.

### Build segmentation artifacts (optional)

You can reproduce the notebook’s segmentation pipeline and generate artifacts under `102025/`:

```powershell
# From the app folder
$base = (Get-Item ..).FullName
python build_segments.py --base $base --stats_xlsx "$base\Stat_KZ092025.xlsx" --regions_path "$base\kz.shp"
```

This will create:
- `regions_enriched.shp` — regions with socio-economic attributes and density/composite
- `segments_fine_heuristic_polygons.geojson` — fine-grained segment polygons
- `segment_code_map.json` — stable mapping from `segment_id` (string) to `segment_code` (int)

At runtime, `predictor.py` will automatically use these if present in `102025/`. You can also override paths via env vars:

- `SEGMENTS_SHP` — path to polygons GeoJSON/SHP
- `STATS_XLSX` — path to `Stat_KZ092025.xlsx`
- `NN_MODEL_DIR` — folder with NN artifacts

## Troubleshooting

- If you see missing column errors, check your file matches the template headers exactly
- If `Stat_KZ092025.xlsx` is missing, copy it into this folder or set `STATS_XLSX` to its path
- To retrain with a different model, re-run the training step and restart Streamlit
- If using GPU-enabled PyTorch, install a CUDA-enabled torch build appropriate for your GPU/driver per the official PyTorch instructions.

## License

MIT (adjust as needed).
