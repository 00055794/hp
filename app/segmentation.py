"""
Segmentation and region enrichment pipeline adapted from the user's notebook cells.

Functions provided:
- create_stats_shapefile: build a shapefile of centroids from Stat_KZ092025.xlsx
- load_regions: load and combine region shapefiles from a folder
- load_regions_from_file: load a single regions shapefile/geojson
- enrich_regions_with_stats: robustly attach socio-economic attributes to regions
- compute_density_and_composite: add counts/area/density/composite_score
- fine_segment_housing: heuristic fine-grained segmentation for house points within regions
- build_segment_code_map: encode segment_id -> integer code and persist mapping

All functions are defensive: optional dependencies are imported lazily and errors are surfaced clearly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def create_stats_shapefile(excel_path: str, shp_path: str) -> None:
    """Create a shapefile of region centroids from the provided Excel.

    Required columns (case-insensitive): latitude, longitude, region,
    srednmes_zarplata, chislennost_naseleniya_092025 (plus many others which
    will be carried through if present).
    """
    try:
        import geopandas as gpd
    except Exception as e:
        raise ImportError("geopandas is required to create shapefiles") from e

    df = pd.read_excel(excel_path)
    # Normalize column names for robustness
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*opts):
        for k in opts:
            if k in cols_lower:
                return cols_lower[k]
        return None

    lat_col = pick("latitude", "lat")
    lon_col = pick("longitude", "lon", "lng")
    if not lat_col or not lon_col:
        raise ValueError("Excel must include latitude and longitude columns")

    gdf = None
    try:
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build GeoDataFrame: {e}")

    # Ensure clean output (remove stale sidecar files)
    shp_path = str(shp_path)
    base, ext = os.path.splitext(shp_path)
    if ext.lower() != ".shp":
        shp_path = base + ".shp"
    for sfx in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = base + sfx
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    gdf.to_file(shp_path)


def load_regions(folder_path: str,
                 target_crs: str = "EPSG:4326",
                 pattern: str = "*.shp",
                 add_source: bool = True,
                 fix_invalid: bool = True,
                 dissolve: bool = False,
                 dissolve_field: Optional[str] = None):
    """Load and combine shapefiles in a folder into a single GeoDataFrame."""
    try:
        import geopandas as gpd
    except Exception as e:
        raise ImportError("geopandas is required to load regions") from e

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")
    shp_files = sorted(folder.glob(pattern))
    if not shp_files:
        raise FileNotFoundError(f"No shapefiles found in {folder} matching pattern {pattern}")
    gdfs: List["gpd.GeoDataFrame"] = []
    for shp in shp_files:
        try:
            gdf = gpd.read_file(shp)
        except Exception as e:
            print(f"[WARN] Failed to read {shp.name}: {e}")
            continue
        if gdf.empty:
            print(f"[INFO] Empty shapefile skipped: {shp.name}")
            continue
        if gdf.crs is None:
            gdf.set_crs(target_crs, inplace=True)
        elif gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        if add_source:
            gdf["source_file"] = shp.stem
        if fix_invalid and not gdf.geometry.is_valid.all():
            gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdfs.append(gdf)
    if not gdfs:
        raise ValueError("All shapefiles failed to load or were empty.")
    combined = pd.concat(gdfs, ignore_index=True, sort=False)
    if dissolve:
        if dissolve_field and dissolve_field in combined.columns:
            combined = combined.dissolve(by=dissolve_field, as_index=False)
        else:
            combined = combined.dissolve().reset_index(drop=True)
    combined.crs = target_crs
    combined = combined[combined.geometry.notna()]
    combined = combined[~combined.geometry.is_empty]
    return combined


def load_regions_from_file(path: str, target_crs: str = "EPSG:4326"):
    try:
        import geopandas as gpd
    except Exception as e:
        raise ImportError("geopandas is required to load regions") from e
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def enrich_regions_with_stats(stats_gdf, regions_gdf,
                              snap_tol_m: float = 250,
                              idw_power: int = 2,
                              k_neighbors: Optional[int] = None):
    """Robust enrichment: spatial join + name fallback + nearest + IDW fill."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from scipy.spatial import cKDTree
    except Exception as e:
        raise ImportError("geopandas and scipy are required for enrichment") from e

    CENTROID_GDF = stats_gdf.copy()
    POLY_GDF = regions_gdf.copy()

    if POLY_GDF.crs is None:
        POLY_GDF = POLY_GDF.set_crs("EPSG:4326")
    if CENTROID_GDF.crs is None:
        CENTROID_GDF = CENTROID_GDF.set_crs(POLY_GDF.crs)
    if POLY_GDF.crs != CENTROID_GDF.crs:
        CENTROID_GDF = CENTROID_GDF.to_crs(POLY_GDF.crs)

    # Work in metric if geographic
    if POLY_GDF.crs.to_string().lower() in ["epsg:4326", "epsg:4258"]:
        POLY_METRIC = POLY_GDF.to_crs(3857)
        CENTROID_METRIC = CENTROID_GDF.to_crs(3857)
    else:
        POLY_METRIC = POLY_GDF
        CENTROID_METRIC = CENTROID_GDF

    within_mask = CENTROID_METRIC.within(POLY_METRIC.unary_union)
    outside_idx = CENTROID_METRIC.index[~within_mask]
    if len(outside_idx) > 0:
        poly_sindex = POLY_METRIC.sindex
        snapped = 0
        for idx in outside_idx:
            pt = CENTROID_METRIC.loc[idx, 'geometry']
            candidates = list(poly_sindex.query(pt.buffer(snap_tol_m)))
            if not candidates:
                continue
            dists = [pt.distance(POLY_METRIC.iloc[i].geometry) for i in candidates]
            j = int(np.argmin(dists))
            if dists[j] <= snap_tol_m:
                CENTROID_METRIC.at[idx, 'geometry'] = POLY_METRIC.iloc[candidates[j]].geometry.representative_point()
                snapped += 1
        if snapped:
            # back to original CRS if needed
            if POLY_GDF.crs.to_string().lower() in ["epsg:4326", "epsg:4258"]:
                CENTROID_GDF['geometry'] = gpd.GeoSeries(CENTROID_METRIC.to_crs(POLY_GDF.crs).geometry, crs=POLY_GDF.crs)
            else:
                CENTROID_GDF = CENTROID_METRIC

    primary_join = gpd.sjoin(CENTROID_GDF, POLY_GDF.reset_index(), how='left', predicate='within')
    # Normalize join index column name to 'poly_index'
    if 'index_right' in primary_join.columns:
        primary_join = primary_join.rename(columns={'index_right': 'poly_index'})
    elif 'poly_index' not in primary_join.columns:
        # Some versions keep the right's original index as a column named 'index'
        guess_cols = [c for c in ['index', 'index_y', 'idx_right', 'right_index'] if c in primary_join.columns]
        if guess_cols:
            primary_join['poly_index'] = primary_join[guess_cols[0]]
        else:
            # create empty and fill via fallbacks
            primary_join['poly_index'] = np.nan

    name_keys = [k for k in ['region','name','NAME','adm_name','ADM_NAME'] if k in CENTROID_GDF.columns and k in POLY_GDF.columns]
    if name_keys:
        key = name_keys[0]
        missing_centroids = primary_join[primary_join['poly_index'].isna()][[key]]
        if not missing_centroids.empty:
            poly_lookup = POLY_GDF.reset_index()[[key, 'index']].drop_duplicates()
            merged = missing_centroids.merge(poly_lookup, on=key, how='left')
            primary_join.loc[primary_join['poly_index'].isna(), 'poly_index'] = merged['index'].values

    still_missing = primary_join['poly_index'].isna()
    if still_missing.any():
        poly_points = POLY_GDF.geometry.representative_point()
        if POLY_GDF.crs.to_string().lower() in ["epsg:4326", "epsg:4258"]:
            poly_pts_proj = poly_points.to_crs(3857)
            cent_proj = primary_join.loc[still_missing, 'geometry'].to_crs(3857)
        else:
            poly_pts_proj = poly_points
            cent_proj = primary_join.loc[still_missing, 'geometry']
        tree = cKDTree(np.vstack([poly_pts_proj.x.values, poly_pts_proj.y.values]).T)
        q_xy = np.vstack([cent_proj.x.values, cent_proj.y.values]).T
        dists, idxs = tree.query(q_xy, k=1)
        primary_join.loc[still_missing, 'poly_index'] = POLY_GDF.iloc[idxs].reset_index()['index'].values

    # Determine attribute columns from centroids, excluding geometry and lat/lon/region-ish
    exclude = {'latitude','longitude','region'}
    attr_cols = [c for c in CENTROID_GDF.columns if c not in exclude and c != 'geometry']
    agg = primary_join.groupby('poly_index')[attr_cols].mean().reset_index()
    POLY_ENRICH = POLY_GDF.reset_index().merge(agg, left_on='index', right_on='poly_index', how='left')

    # IDW fill if needed
    missing_any = POLY_ENRICH[attr_cols].isna().any(axis=1)
    if missing_any.any():
        if POLY_GDF.crs.to_string().lower() in ["epsg:4326", "epsg:4258"]:
            cent_proj_all = CENTROID_GDF.to_crs(3857)
            poly_centroids_proj = POLY_ENRICH.loc[missing_any, 'geometry'].to_crs(3857).centroid
        else:
            cent_proj_all = CENTROID_GDF
            poly_centroids_proj = POLY_ENRICH.loc[missing_any, 'geometry'].centroid
        centroid_xy = np.vstack([cent_proj_all.geometry.x.values, cent_proj_all.geometry.y.values]).T
        tree2 = cKDTree(centroid_xy)
        target_xy = np.vstack([poly_centroids_proj.x.values, poly_centroids_proj.y.values]).T
        k = min(8, len(centroid_xy)) if k_neighbors is None else int(k_neighbors)
        if k <= 0:
            return POLY_ENRICH
        dists, idxs = tree2.query(target_xy, k=k)
        if k == 1:
            dists = dists[:, None]; idxs = idxs[:, None]
        dists = np.where(dists == 0, 1e-6, dists)
        w = 1 / (dists ** idw_power)
        w = w / w.sum(axis=1, keepdims=True)
        miss_idx = POLY_ENRICH.index[missing_any]
        for col in attr_cols:
            src_vals = CENTROID_GDF[col].values
            interp_vals = (w * src_vals[idxs]).sum(axis=1)
            need = POLY_ENRICH.loc[miss_idx, col].isna()
            POLY_ENRICH.loc[miss_idx[need], col] = interp_vals[need.values]
    return POLY_ENRICH


def compute_density_and_composite(regions_enriched, hp_points_gdf=None) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except Exception as e:
        raise ImportError("geopandas is required for density/composite computations") from e

    regions = regions_enriched.copy()
    # Counts per region
    if hp_points_gdf is not None and not hp_points_gdf.empty:
        hp_join = gpd.sjoin(hp_points_gdf, regions[['index','geometry']], how='left', predicate='within')
        counts = hp_join.groupby('index').size().rename('house_point_count')
        regions = regions.join(counts, on='index')
    else:
        regions['house_point_count'] = 0
    # Area and density
    regions_proj = regions.to_crs('EPSG:3857')
    regions['area_km2'] = regions_proj.geometry.area / 1e6
    regions['density_hp_per_km2'] = regions['house_point_count'] / regions['area_km2'].replace({0: np.nan})
    regions['density_hp_per_km2'] = regions['density_hp_per_km2'].fillna(0)

    # Composite score using ordered attributes if present
    features = [
        'srednmes_zarplata',
        'index_real_zarplaty',
        'chislennost_naseleniya_092025',
        'prirost_naselenya',
        'temp_prirosta_percent',
        'index_potreb_cen_tovary_uslugi',
        'index_potreb_cen_prodovolstv_tovary',
        'index_potreb_cen_neprodovolstv_tovary',
        'index_potreb_cen_platnye_uslugi',
    ]
    features = [c for c in features if c in regions.columns]
    if features:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_std = scaler.fit_transform(regions[features].values)
        alpha = 0.85
        weights = np.array([alpha ** i for i in range(len(features))])
        weights = weights / weights.sum()
        X_weighted = X_std * weights
        regions['composite_score'] = X_weighted.sum(axis=1)
    else:
        regions['composite_score'] = 0.0
    return regions


def fine_segment_housing(
    hp_gdf,
    regions_gdf,
    region_key_candidates: Iterable[str] = ("id", "poly_index", "code"),
    region_attr_cols: Iterable[str] = (
        "chislennost_naseleniya_092025",
        "srednmes_zarplata",
        "temp_prirosta_percent",
        "house_point_count",
        "area_km2",
        "density_hp_per_km2",
    ),
    target_total_segments: Optional[int] = None,
    k_factor: float = 1.15,
    desired_avg_points: Optional[int] = 340,
    alpha_exp: float = 0.8,
    min_points_to_split: int = 50,
    max_segments_region: int = 1200,
    min_segments_region: int = 1,
    geo_weight: float = 1.0,
    jitter_scale: float = 0.0,
    use_metric_crs: bool = True,
    metric_crs: str = "EPSG:3857",
    buffer_radius_m: float = 1800.0,
    hull_min_points: int = 5,
    enable_second_pass: bool = True,
    second_pass_max_points: int = 850,
    second_pass_target_avg: int = 420,
    second_pass_max_new_segments: int = 1200,
    enforce_non_overlap: bool = True,
    non_overlap_order: str = 'points',
    min_area_m2: float = 0.0,
    random_state: int = 42,
    batch_size: int = 4096,
):
    """Heuristic fine segmentation largely mirroring the notebook implementation."""
    import geopandas as gpd
    from shapely.ops import unary_union
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans

    if hp_gdf.empty:
        raise ValueError("hp_gdf is empty")
    if regions_gdf.empty:
        raise ValueError("regions_gdf is empty")

    points = hp_gdf.copy()
    regions = regions_gdf.copy()

    region_key = None
    for cand in region_key_candidates:
        if cand in regions.columns:
            region_key = cand
            break
    if region_key is None:
        regions = regions.reset_index().rename(columns={"index": "region_key"})
        region_key = "region_key"

    if regions.crs is None:
        raise ValueError("regions_gdf CRS is None; set it before calling.")
    if points.crs is None:
        points = points.set_crs(regions.crs)
    if points.crs != regions.crs:
        points = points.to_crs(regions.crs)

    base_crs = regions.crs

    # Spatial join
    joined = gpd.sjoin(points, regions[[region_key, 'geometry'] + [c for c in region_attr_cols if c in regions.columns]], how='left', predicate='within')
    miss = joined[region_key].isna().sum()
    if miss:
        joined = joined.dropna(subset=[region_key])
    N = len(joined)
    if N == 0:
        raise ValueError("No points after join.")

    # Heuristic segments target
    if target_total_segments is None:
        h1 = k_factor * np.sqrt(N)
        h2 = N / desired_avg_points if desired_avg_points else 0
        target_total_segments = int(max(h1, h2))
    target_total_segments = max(target_total_segments, 21)
    target_total_segments = min(target_total_segments, 4572)

    region_counts = joined[region_key].value_counts().rename('n')
    weights = region_counts.pow(alpha_exp)
    alloc_float = weights / weights.sum() * target_total_segments
    alloc = pd.DataFrame({region_key: alloc_float.index, 'alloc_float': alloc_float.values, 'n': region_counts.values})
    alloc['k_region'] = alloc['alloc_float'].apply(lambda v: max(min_segments_region, int(np.floor(v))))
    alloc.loc[alloc['n'] < min_points_to_split, 'k_region'] = 1
    alloc['k_region'] = alloc['k_region'].clip(upper=max_segments_region)

    def adjust(df, target):
        diff = df['k_region'].sum() - target
        if diff == 0:
            return df
        while diff > 0:
            idxs = df[df['k_region'] > 1].sort_values('k_region', ascending=False).index
            if not len(idxs):
                break
            df.at[idxs[0], 'k_region'] -= 1
            diff -= 1
        while diff < 0:
            idxs = df.sort_values('n', ascending=False).index
            for i in idxs:
                df.at[i, 'k_region'] += 1
                diff += 1
                if diff == 0:
                    break
        return df
    alloc = adjust(alloc, target_total_segments)

    # Build feature matrix
    attr_cols_avail = [c for c in region_attr_cols if c in joined.columns]
    attr_block = joined[attr_cols_avail].copy().apply(lambda s: s.fillna(s.median())) if attr_cols_avail else pd.DataFrame(index=joined.index)
    attr_scaled = StandardScaler().fit_transform(attr_block) if attr_cols_avail else np.empty((len(joined), 0))
    geo_xy = np.vstack([joined.geometry.x.values, joined.geometry.y.values]).T
    if jitter_scale > 0:
        geo_xy = geo_xy + np.random.normal(scale=jitter_scale, size=geo_xy.shape)
    geo_scaled = StandardScaler().fit_transform(geo_xy) * geo_weight
    X = np.hstack([geo_scaled, attr_scaled])
    feat_cols = [f'f_{i}' for i in range(X.shape[1])]
    for i, c in enumerate(feat_cols):
        joined[c] = X[:, i]

    # Region-wise clustering
    from sklearn.cluster import MiniBatchKMeans
    joined['segment_local'] = -1
    cluster_offset = 0
    for _, row in alloc.iterrows():
        rid = row[region_key]
        k_r = int(row['k_region'])
        idx = joined[joined[region_key] == rid].index
        if not len(idx):
            continue
        if k_r <= 1:
            joined.loc[idx, 'segment_local'] = cluster_offset
            cluster_offset += 1
            continue
        subX = joined.loc[idx, feat_cols].values
        try:
            km = MiniBatchKMeans(n_clusters=k_r, random_state=random_state, batch_size=batch_size)
            labels = km.fit_predict(subX)
            joined.loc[idx, 'segment_local'] = labels + cluster_offset
            cluster_offset += k_r
        except Exception as e:
            joined.loc[idx, 'segment_local'] = cluster_offset
            cluster_offset += 1

    joined['segment_id'] = joined.apply(lambda r: f"R{r[region_key]}-S{int(r['segment_local'])}", axis=1)

    # Polygonization in metric CRS
    region_geom_map = regions.set_index(region_key).geometry
    points_metric = joined
    region_geom_metric = region_geom_map
    work_crs = base_crs
    if use_metric_crs and str(base_crs) != metric_crs:
        points_metric = joined.to_crs(metric_crs)
        region_geom_metric = region_geom_map.to_crs(metric_crs)
        work_crs = metric_crs

    seg_polys = []
    for seg_id, grp in points_metric.groupby('segment_id'):
        rid = grp.iloc[0][region_key]
        region_geom = region_geom_metric.get(rid)
        pts = grp.geometry
        if len(pts) >= hull_min_points:
            hull = pts.unary_union.convex_hull.buffer(buffer_radius_m)
        else:
            hull = unary_union([p.buffer(buffer_radius_m * 0.6) for p in pts])
        if region_geom is not None:
            try:
                hull = hull.intersection(region_geom)
            except Exception:
                pass
        seg_polys.append({'segment_id': seg_id, region_key: rid, 'geometry': hull})

    segments_gdf = gpd.GeoDataFrame(seg_polys, geometry='geometry', crs=work_crs)

    # Enforce non-overlap greedily if requested (work in metric CRS)
    if enforce_non_overlap and len(segments_gdf):
        segs_ord = segments_gdf.copy()
        # Build ordering by points count
        order_series = joined.groupby('segment_id').size().rename('order_val')
        segs_ord = segs_ord.merge(order_series.to_frame(), on='segment_id', how='left')
        segs_ord['order_val'] = segs_ord['order_val'].fillna(1)
        segs_ord = segs_ord.sort_values('order_val', ascending=False).reset_index(drop=True)
        occupied = None
        cleaned_geoms = []
        for geom in segs_ord.geometry:
            cleaned = geom
            if occupied is not None:
                try:
                    cleaned = geom.difference(occupied)
                except Exception:
                    cleaned = geom
            if (cleaned is not None) and (not cleaned.is_empty):
                occupied = cleaned if occupied is None else occupied.union(cleaned)
            cleaned_geoms.append(cleaned)
        segs_ord['geometry'] = cleaned_geoms
        if (min_area_m2 and min_area_m2 > 0) and work_crs == metric_crs:
            segs_ord['__area'] = segs_ord.geometry.area
            segs_ord.loc[segs_ord['__area'] < min_area_m2, 'geometry'] = None
            segs_ord = segs_ord.drop(columns=['__area'])
        segments_gdf = segs_ord.drop(columns=['order_val'])

    if work_crs != base_crs:
        segments_gdf = segments_gdf.to_crs(base_crs)

    # Summary per segment
    summary_rows = []
    for seg_id, grp in joined.groupby('segment_id'):
        rec = {
            'segment_id': seg_id,
            'points': len(grp),
            region_key: grp.iloc[0][region_key],
        }
        summary_rows.append(rec)
    segments_summary = pd.DataFrame(summary_rows).sort_values('points', ascending=False)
    return joined, segments_gdf, segments_summary


def build_segment_code_map(segment_ids: Iterable[str]) -> dict:
    """Return a stable mapping dict segment_id -> code (int) using sorted unique order."""
    uniq = sorted({str(s) for s in segment_ids})
    return {s: i for i, s in enumerate(uniq)}

