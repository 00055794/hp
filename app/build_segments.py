from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from segmentation import (
    create_stats_shapefile,
    load_regions_from_file,
    enrich_regions_with_stats,
    compute_density_and_composite,
    fine_segment_housing,
    build_segment_code_map,
)


def _resolve_under_base(base: Path, p: Path) -> Path:
    if p.exists():
        return p
    candidate = base / p.name
    if candidate.exists():
        return candidate
    return p


def main(base_dir: str, stats_xlsx: str, regions_path: str, hp_points_csv: str | None = None, hp_file: str | None = None):
    import geopandas as gpd

    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    stats_xlsx_path = _resolve_under_base(base, Path(stats_xlsx))
    regions_path_path = _resolve_under_base(base, Path(regions_path))
    if not stats_xlsx_path.exists():
        raise FileNotFoundError(f"Stats Excel not found: {stats_xlsx!r} (also tried {str(base / Path(stats_xlsx).name)!r})")
    if not regions_path_path.exists():
        raise FileNotFoundError(f"Regions file not found: {regions_path!r} (also tried {str(base / Path(regions_path).name)!r})")

    stats_shp = base / "stat.shp"
    if not stats_shp.exists():
        print("Creating stats shapefile from Excel…")
        create_stats_shapefile(str(stats_xlsx_path), str(stats_shp))
    else:
        print("Using existing stats shapefile:", stats_shp)

    print("Loading regions…")
    regions = load_regions_from_file(str(regions_path_path))
    stats_gdf = gpd.read_file(stats_shp)

    print("Enriching regions with socio-economic attributes…")
    regions_enriched = enrich_regions_with_stats(stats_gdf, regions)
    regions_enriched = compute_density_and_composite(regions_enriched)

    out_regions = base / "regions_enriched.shp"
    print("Saving:", out_regions)
    for sfx in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = out_regions.with_suffix(sfx)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    regions_enriched.to_file(out_regions)

    # Optional: points GeoDataFrame from file (CSV/XLSX/GeoJSON/Parquet/Shapefile)
    hp_gdf = None
    def _cols(d):
        return {c.lower(): c for c in d.columns}
    def _pick(cols_map, *names):
        for n in names:
            if n in cols_map:
                return cols_map[n]
        return None
    points_path = None
    if hp_file:
        points_path = _resolve_under_base(base, Path(hp_file))
    elif hp_points_csv:
        points_path = _resolve_under_base(base, Path(hp_points_csv))
    if points_path and points_path.exists():
        ext = points_path.suffix.lower()
        try:
            if ext in [".geojson", ".json", ".gpkg", ".shp", ".gdb", ".fgb"]:
                print(f"Loading house points (geospatial): {points_path.name}")
                hp_gdf = gpd.read_file(points_path)
                if hp_gdf.crs is None:
                    hp_gdf = hp_gdf.set_crs("EPSG:4326")
                elif hp_gdf.crs.to_string() != "EPSG:4326":
                    hp_gdf = hp_gdf.to_crs("EPSG:4326")
            elif ext in [".parquet", ".pq"]:
                print(f"Loading house points (parquet): {points_path.name}")
                # geopandas.read_parquet preserves geometry if present
                hp_gdf = gpd.read_parquet(points_path)
                if hp_gdf.crs is None:
                    hp_gdf = hp_gdf.set_crs("EPSG:4326")
                elif hp_gdf.crs.to_string() != "EPSG:4326":
                    hp_gdf = hp_gdf.to_crs("EPSG:4326")
            elif ext in [".csv", ".txt", ".tsv", ".xlsx", ".xls"]:
                print(f"Loading house points (tabular): {points_path.name}")
                if ext in [".xlsx", ".xls"]:
                    df_pts = pd.read_excel(points_path)
                else:
                    df_pts = pd.read_csv(points_path)
                cm = _cols(df_pts)
                lat_col = _pick(cm, "latitude", "lat", "y")
                lon_col = _pick(cm, "longitude", "lon", "lng", "x")
                if lat_col and lon_col:
                    hp_gdf = gpd.GeoDataFrame(df_pts.copy(), geometry=gpd.points_from_xy(df_pts[lon_col], df_pts[lat_col]), crs="EPSG:4326")
                else:
                    print("Tabular file missing latitude/longitude columns; skipping segmentation.")
            else:
                print(f"Unsupported points file extension: {ext}")
        except Exception as e:
            print(f"[WARN] Failed to load house points: {e}")

    if hp_gdf is not None and not hp_gdf.empty:
        print("Running fine-grained segmentation…")
        pts_with_segments, seg_polys, seg_summary = fine_segment_housing(hp_gdf, regions_enriched)
        print(f"Segments: {len(seg_summary)}")
        seg_polys_path = base / "segments_fine_heuristic_polygons.geojson"
        seg_pts_path = base / "segments_fine_heuristic_points.parquet"
        seg_sum_path = base / "segments_fine_heuristic_summary.csv"
        seg_polys.to_file(seg_polys_path, driver="GeoJSON")
        pts_with_segments[["segment_id", "segment_local"]].to_parquet(seg_pts_path, index=False)
        seg_summary.to_csv(seg_sum_path, index=False)

        # Build stable code map
        code_map = build_segment_code_map(pts_with_segments["segment_id"].unique().tolist())
        with open(base / "segment_code_map.json", "w", encoding="utf-8") as f:
            json.dump(code_map, f, ensure_ascii=False)
        print("Saved segment_code_map.json with", len(code_map), "entries")
    else:
        print("No house points provided; skipped fine segmentation. You can still use regions_enriched for stats.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parents[1]), help="Base directory to place outputs (default: repo/102025)")
    ap.add_argument("--stats_xlsx", default="Stat_KZ092025.xlsx", help="Path to Stat_KZ092025.xlsx")
    ap.add_argument("--regions_path", default="kz.shp", help="Shapefile/GeoJSON for regions (default: kz.shp)")
    ap.add_argument("--hp_csv", default=None, help="Optional house points CSV with LATITUDE/LONGITUDE")
    ap.add_argument("--hp_file", default=None, help="Optional house points file (CSV/XLSX/GeoJSON/Parquet/Shapefile)")
    args = ap.parse_args()
    main(args.base, args.stats_xlsx, args.regions_path, hp_points_csv=args.hp_csv, hp_file=args.hp_file)
