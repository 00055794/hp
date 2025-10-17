# Deployment Guide

## Reducing Project Size for GitHub and Render.com

### Files Excluded from Git (via .gitignore)
- `.venv/` - Virtual environment (1.2GB)
- `__pycache__/` - Python cache files
- `.ipynb_checkpoints` - Jupyter checkpoints
- IDE configuration files

### Files Excluded from Render Deployment (via .slugignore)
- Documentation files (*.md, INFO.txt)
- Development notebooks (*.ipynb)
- Training scripts (train_model.py, train_nn.py)
- Scraping tools (scrape_krisha.py, krisha_parser.py)
- Build scripts (make_zip.ps1, build_segments.py)
- Sample files (template.csv)

### Project Structure for Deployment
```
102025_3/
├── app/
│   ├── app.py              # Main Streamlit app
│   ├── predictor.py        # Inference logic
│   ├── requirements.txt    # Dependencies
│   ├── nn_model/          # Pre-trained model files
│   │   ├── model.pt
│   │   ├── scaler_X.joblib
│   │   ├── scaler_y.joblib
│   │   ├── cat_mappings.json
│   │   └── feature_list.json
├── segments_fine_heuristic_polygons.geojson
├── segments_fine_heuristic_summary.csv
├── kz.* (shapefiles)
├── regions_enriched.* (shapefiles)
└── stat.* (shapefiles)
```

## GitHub Setup

1. Initialize git repository (if not already):
```powershell
cd c:\Users\00055794\Desktop\Gulnaz\PROJECTS\House_Prices\102025_3
git init
git add .
git commit -m "Initial commit - optimized for deployment"
```

2. Connect to GitHub:
```powershell
git remote add origin https://github.com/andprov/krisha.kz.git
git branch -M main
git push -u origin main
```

## Render.com Deployment

### Option 1: Web Service (Recommended)
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Configure:
   - **Build Command**: `pip install -r app/requirements.txt`
   - **Start Command**: `streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment**: Python 3

### Option 2: Using render.yaml
Create a `render.yaml` file in the root directory (see below).

## Estimated Sizes
- **Before optimization**: ~1.2 GB (with .venv)
- **After optimization (Git)**: ~5-10 MB
- **Deployed slug (Render)**: ~3-5 MB (excluding dependencies)

## Notes
- The `.venv` directory won't be pushed to GitHub
- Render will install dependencies from `requirements.txt` during deployment
- Model files in `nn_model/` are small (~100KB total) and are included
- Geospatial files are necessary for the app functionality
