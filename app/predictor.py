import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load


KZ_CENTER = (48.0196, 66.9237)  # lat, lon


@dataclass
class RegionStatsIndex:
    # KD-tree like index built from stats centroids for fast nearest lookup
    names: List[str]
    ids: List[str]
    lats: np.ndarray
    lons: np.ndarray
    srednmes: np.ndarray
    chisl: np.ndarray

    @classmethod
    def from_excel(cls, xlsx_path: str) -> "RegionStatsIndex":
        df = pd.read_excel(xlsx_path)
        # try sensible column names; allow flexible casing/variants
        cols = {c.lower(): c for c in df.columns}
        def pick(*options):
            for k in options:
                if k in cols:
                    return cols[k]
            return None

        lat_col = pick("latitude", "lat")
        lon_col = pick("longitude", "lon", "lng")
        name_col = pick("name", "region", "region_name")
        id_col = pick("id", "poly_index", "region_id")
        zarplata_col = pick("srednmes_zarplata", "avg_salary", "srednaya_zp")
        chisl_col = pick("chislennost_naseleniya_092025", "population", "chislennost")

        if not (lat_col and lon_col and zarplata_col and chisl_col):
            raise ValueError(
                "Stats Excel must contain latitude, longitude, srednmes_zarplata, chislennost_naseleniya_092025 columns."
            )

        names = df[name_col].astype(str).fillna("") if name_col else pd.Series([""] * len(df))
        ids = df[id_col].astype(str).fillna("") if id_col else pd.Series([""] * len(df))
        return cls(
            names=names.tolist(),
            ids=ids.tolist(),
            lats=df[lat_col].astype(float).to_numpy(),
            lons=df[lon_col].astype(float).to_numpy(),
            srednmes=df[zarplata_col].astype(float).to_numpy(),
            chisl=df[chisl_col].astype(float).to_numpy(),
        )

    def nearest(self, lat: float, lon: float) -> Tuple[str, str, float, float]:
        # great-circle approximation using vectorized haversine
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        lat2 = np.radians(self.lats)
        lon2 = np.radians(self.lons)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        d = 2 * np.arcsin(np.sqrt(a))  # radians
        idx = int(np.nanargmin(d)) if len(d) else 0
        seg_code = self.ids[idx] if self.ids[idx] else self.names[idx]
        return seg_code, self.names[idx], float(self.srednmes[idx]), float(self.chisl[idx])


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Accept both 'TOTAL AREA' and 'TOTAL_AREA'; unify to expected names used in training pipeline.
    rename_map = {
        "TOTAL AREA": "TOTAL_AREA",
        "Latitude": "LATITUDE",
        "Longitude": "LONGITUDE",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df


REQUIRED_FEATURES = [
    "ROOMS",
    "TOTAL_AREA",
    "FLOOR",
    "TOTAL_FLOORS",
    "FURNITURE",
    "CONDITION",
    "CEILING",
    "MATERIAL",
    "YEAR",
    "LATITUDE",
    "LONGITUDE",
]


def attach_region_features(df: pd.DataFrame, stats_index: RegionStatsIndex) -> pd.DataFrame:
    df = df.copy()
    segs = []
    regions = []
    sredn = []
    chisl = []
    for lat, lon in zip(df["LATITUDE"].astype(float), df["LONGITUDE"].astype(float)):
        try:
            seg, name, s, c = stats_index.nearest(lat, lon)
        except Exception:
            seg, name, s, c = ("", "", np.nan, np.nan)
        segs.append(seg)
        regions.append(name)
        sredn.append(s)
        chisl.append(c)
    df["segment_code"] = segs
    df["region_name"] = regions
    df["srednmes_zarplata"] = sredn
    df["chislennost_naseleniya_092025"] = chisl
    # Optionally override segment_code using a shapefile if available
    df = _maybe_assign_segment_from_shapefile(df)
    # Ensure numeric code
    if "segment_code" in df.columns:
        try:
            df["segment_code"] = pd.to_numeric(df["segment_code"], errors="coerce").fillna(-1).astype(int)
        except Exception:
            df["segment_code"] = -1
    return df


def _maybe_assign_segment_from_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """If SEGMENTS_SHP env var (or default known path) is present and geopandas is installed,
    perform a spatial join to assign a more precise segment_code as in the notebook segmentation.
    """
    shp = os.environ.get("SEGMENTS_SHP", "")
    if not shp:
        # Try common locations
        parent = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        cand = [
            os.path.join(parent, "segments_fine_heuristic_polygons.geojson"),
            os.path.join(parent, "segments_spatial_constrained.shp"),
            os.path.join(parent, "regions_enriched.shp"),
            os.path.join(parent, "kz.shp"),
        ]
        for p in cand:
            if os.path.exists(p):
                shp = p
                break
    if not shp or not os.path.exists(shp):
        return df
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception:
        return df
    try:
        seg_gdf = gpd.read_file(shp)
        if seg_gdf.crs is None:
            seg_gdf.set_crs("EPSG:4326", inplace=True)
        elif seg_gdf.crs.to_string() != "EPSG:4326":
            seg_gdf = seg_gdf.to_crs("EPSG:4326")
        pts = gpd.GeoDataFrame(
            df.reset_index(drop=True).copy(),
            geometry=gpd.points_from_xy(df["LONGITUDE"].astype(float), df["LATITUDE"].astype(float)),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, seg_gdf, how="left", predicate="within")
        # Try to pick a reasonable segment id/code column
        cand_cols = [
            "segment_id",
            "segment_code",
            "segment",
            "id",
        ]
        seg_col = next((c for c in cand_cols if c in joined.columns), None)
        parent = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        map_json = os.path.join(parent, "segment_code_map.json")
        mapping = None
        if os.path.exists(map_json):
            try:
                with open(map_json, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
            except Exception:
                mapping = None
        if seg_col and mapping and seg_col == "segment_id":
            # Stable mapping produced during segmentation build
            df["segment_code"] = joined[seg_col].map(mapping).fillna(df.get("segment_code", -1)).astype(int)
        elif seg_col:
            # If a numeric code column exists in polygons, use it directly; otherwise fallback to string
            try:
                df["segment_code"] = pd.to_numeric(joined[seg_col], errors="coerce").fillna(df.get("segment_code", -1)).astype(int)
            except Exception:
                # Build stable mapping from polygons themselves (LabelEncoder-style)
                if seg_col == "segment_id":
                    uniq_ids = sorted([str(v) for v in seg_gdf[seg_col].dropna().unique().tolist()])
                    local_map = {s: i for i, s in enumerate(uniq_ids)}
                    df["segment_code"] = joined[seg_col].map(local_map).fillna(df.get("segment_code", -1)).astype(int)
                else:
                    vals = joined[seg_col].astype(str).fillna("")
                    uniq = {v: i for i, v in enumerate(sorted(set([v for v in vals if v])))}
                    df["segment_code"] = [uniq.get(v, -1) for v in vals]
        # Optionally set region_name if available
        name_col = next((c for c in ["NAME", "NAME_1", "region_name", "region"] if c in joined.columns), None)
        if name_col and "region_name" in df.columns:
            df["region_name"] = joined[name_col].astype(str).fillna(df["region_name"])
        return df
    except Exception:
        return df


def load_pipeline(model_path: str):
    return load(model_path)


def predict_kzt(pipeline, input_df: pd.DataFrame) -> np.ndarray:
    # The model is trained to predict ln(price); convert to real KZT
    ln_pred = pipeline.predict(input_df)
    # handle any potential scaling: if the regressor predicts ln directly, exp; if something else, still exp
    return np.exp(ln_pred)


def validate_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_input_columns(df)
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    return df, missing


def default_map_center(rows: pd.DataFrame) -> Tuple[float, float]:
    if len(rows) and {"LATITUDE", "LONGITUDE"}.issubset(rows.columns):
        return float(rows["LATITUDE"].mean()), float(rows["LONGITUDE"].mean())
    return KZ_CENTER
