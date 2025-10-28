"""
Streamlit Web Application for KZ Real Estate Price Prediction
Uses the complete pipeline that exactly matches notebook training.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
try:
    from folium.plugins import MousePosition, Geocoder
except ImportError:
    from folium.plugins import MousePosition
    Geocoder = None
import traceback

# Import our notebook-matched pipeline
from pipeline_complete import CompletePipeline

st.set_page_config(page_title="KZ Real Estate Price Estimator", layout="wide")
st.title("KZ Real Estate Price Estimator")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the complete prediction pipeline once"""
    try:
        # Get the directory where this script is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(app_dir)
        
        pipeline = CompletePipeline(
            model_dir=os.path.join(app_dir, "nn_model"),
            region_grid_lookup=os.path.join(app_dir, "region_grid_lookup.json"),
            region_grid_encoder=os.path.join(app_dir, "region_grid_encoder.json"),
            segments_geojson=os.path.join(parent_dir, "segments_fine_heuristic_polygons.geojson")
        )
        return pipeline, None
    except Exception as e:
        return None, f"Failed to load pipeline: {str(e)}\n{traceback.format_exc()}"


# Load pipeline
pipeline, error = load_pipeline()

if error:
    st.error(f"Pipeline Load Error: {error}")
    st.stop()

# Two-column layout: Single Prediction | Batch Upload
col_single, col_batch = st.columns([1, 1])

# ==================== LEFT: Single Prediction ====================
with col_single:
    st.subheader("Single Prediction")
    
    # Map picker (collapsed by default)
    with st.expander("Pick location on map", expanded=False):
        # Use session state for coordinates
        if "LATITUDE" not in st.session_state:
            st.session_state["LATITUDE"] = 43.2567
        if "LONGITUDE" not in st.session_state:
            st.session_state["LONGITUDE"] = 76.9286
        
        pick_lat = st.session_state.get("LATITUDE", 43.2567)
        pick_lon = st.session_state.get("LONGITUDE", 76.9286)
        
        m = folium.Map(location=[pick_lat, pick_lon], zoom_start=12, tiles=None)
        
        # Add satellite and OSM layers
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True,
            show=True
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="OSM", overlay=False, control=True, show=False).add_to(m)
        
        # Add mouse position and click popup
        MousePosition(position="topright", prefix="Lat/Lon:", separator=", ", num_digits=6).add_to(m)
        folium.LatLngPopup().add_to(m)
        
        # Add geocoder search if available (compact, near zoom controls)
        if Geocoder is not None:
            try:
                Geocoder(
                    collapsed=True,
                    position="topleft",
                    placeholder="Search...",
                    add_marker=True
                ).add_to(m)
            except Exception:
                pass
        
        # Add marker at current location
        folium.Marker(
            [pick_lat, pick_lon],
            popup=f"{pick_lat:.6f}, {pick_lon:.6f}",
            tooltip="Current location"
        ).add_to(m)
        
        folium.LayerControl(collapsed=True).add_to(m)
        
        # Display map and capture clicks
        map_data = st_folium(m, height=350, use_container_width=True, key="map_picker")
        
        # Update coordinates from map click
        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            if clicked_lat and clicked_lon:
                st.session_state["LATITUDE"] = clicked_lat
                st.session_state["LONGITUDE"] = clicked_lon
                st.success(f"Selected: {clicked_lat:.6f}, {clicked_lon:.6f}")
    
    # Compact form layout - 3 rows with multiple columns
    # Row 1: Location and Size
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    LATITUDE = r1c1.number_input("Latitude", 40.0, 55.0, st.session_state.get("LATITUDE", 43.2567), format="%.6f")
    LONGITUDE = r1c2.number_input("Longitude", 46.0, 87.0, st.session_state.get("LONGITUDE", 76.9286), format="%.6f")
    ROOMS = r1c3.number_input("Rooms", 1, 10, 2)
    TOTAL_AREA = r1c4.number_input("Area (m²)", 10.0, 500.0, 62.0, 1.0)
    
    # Row 2: Building details (5 columns to fit Floor/Total separately)
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    FLOOR = r2c1.number_input("Floor", 1, 100, 5)
    TOTAL_FLOORS = r2c2.number_input("Total floors", 1, 100, 9)
    YEAR = r2c3.number_input("Year", 1950, 2025, 2015)
    CEILING = r2c4.number_input("Ceiling (m)", 2.0, 5.0, 2.7, 0.1)
    MATERIAL = r2c5.selectbox("Material", [1, 2, 3, 4], 1, format_func=lambda x: {1: "Brick", 2: "Panel", 3: "Monolith", 4: "Other"}[x])
    
    # Row 3: Quality
    r3c1, r3c2 = st.columns(2)
    FURNITURE = r3c1.selectbox("Furniture", [1, 2, 3], 1, format_func=lambda x: {1: "No", 2: "Partial", 3: "Full"}[x])
    CONDITION = r3c2.selectbox("Condition", [1, 2, 3, 4, 5], 2, format_func=lambda x: f"{x} - {'Poor' if x==1 else 'Fair' if x==2 else 'Good' if x==3 else 'Excellent' if x==4 else 'Perfect'}")
    
    # Predict button
    if st.button("Predict Price", use_container_width=True):
        try:
            # Prepare input
            input_data = {
                'ROOMS': ROOMS,
                'LONGITUDE': LONGITUDE,
                'LATITUDE': LATITUDE,
                'TOTAL_AREA': TOTAL_AREA,
                'FLOOR': FLOOR,
                'TOTAL_FLOORS': TOTAL_FLOORS,
                'FURNITURE': FURNITURE,
                'CONDITION': CONDITION,
                'CEILING': CEILING,
                'MATERIAL': MATERIAL,
                'YEAR': YEAR
            }
            
            # Make prediction
            price_kzt, features_df = pipeline.predict_single(input_data, return_features=True)
            
            # Display result
            st.success(f"Predicted Price: {price_kzt:,.0f} KZT")
            st.metric("Price per m²", f"{price_kzt/TOTAL_AREA:,.0f} KZT/m²")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ==================== RIGHT: Batch Upload ====================
with col_batch:
    st.subheader("Batch Upload")
    
    upload = st.file_uploader("Upload CSV file", type=["csv"], help="CSV with required columns")
    
    # Download previous predictions
    b_csv = st.session_state.get("batch_csv_bytes")
    b_csv_name = st.session_state.get("batch_csv_name", "predictions.csv")
    
    if b_csv:
        st.download_button(
            label="Download predictions",
            data=b_csv,
            file_name=b_csv_name,
            mime="text/csv",
            use_container_width=True
        )
    
    if upload is not None:
        try:
            # Read CSV with full float precision for lat/lon
            df_in = pd.read_csv(upload, float_precision='high')
            
            # Ensure lat/lon have full precision (not rounded)
            if 'LATITUDE' in df_in.columns:
                df_in['LATITUDE'] = df_in['LATITUDE'].astype(float)
            if 'LONGITUDE' in df_in.columns:
                df_in['LONGITUDE'] = df_in['LONGITUDE'].astype(float)
            
            # Make predictions
            predictions_kzt = pipeline.predict_batch(df_in)
            
            # Add predictions to dataframe
            df_out = df_in.copy()
            df_out['pred_price_kzt'] = predictions_kzt
            
            st.success(f"Predicted {len(df_out)} properties")
            
            # Show results with all input features + prediction
            display_cols = [c for c in ["ROOMS", "LONGITUDE", "LATITUDE", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION", "CEILING", "MATERIAL", "YEAR", "pred_price_kzt"] if c in df_out.columns]
            
            # Format display to show full lat/lon precision
            df_display = df_out[display_cols].head(50).copy()
            st.dataframe(df_display, use_container_width=True)
            
            # Prepare download with full precision
            import time
            csv_bytes = df_out.to_csv(index=False, float_format='%.10f').encode("utf-8")
            st.session_state["batch_csv_bytes"] = csv_bytes
            _ts = time.strftime("%Y%m%d_%H%M%S")
            st.session_state["batch_csv_name"] = f"predictions_{_ts}.csv"
            
        except Exception as e:
            st.error(f"Batch prediction failed: {str(e)}")
    
    # Template download
    with st.expander("Download template"):
        template_data = pd.DataFrame([{
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
        }])
        
        csv_template = template_data.to_csv(index=False)
        st.download_button(
            label="Download CSV template",
            data=csv_template,
            file_name="template.csv",
            mime="text/csv",
            use_container_width=True
        )
