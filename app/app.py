import os
import io
import uuid
import streamlit as st
import pandas as pd
import folium
try:
    from folium.plugins import MarkerCluster, MousePosition, Geocoder as _FoliumGeocoder, Fullscreen as _Fullscreen
except Exception:
    from folium.plugins import MarkerCluster, MousePosition
    _FoliumGeocoder = None
    _Fullscreen = None
from streamlit_folium import st_folium
import requests
import time
import re

from predictor import (
    RegionStatsIndex,
    attach_region_features,
    load_pipeline,
    predict_kzt,
    validate_inputs,
    REQUIRED_FEATURES,
    default_map_center,
)
from nn_inference import (
    load_artifacts as load_nn_artifacts,
    predict_prices_kzt as nn_predict_kzt,
    apply_cat_maps as nn_apply_cat_maps,
)


st.set_page_config(page_title="KZ Real Estate Price Estimator", layout="wide")
st.title("KZ Real Estate Price Estimator")
st.caption("Predict apartment prices in KZT and view them on the map. Upload your file or use the form.")


@st.cache_resource(show_spinner=False)
def get_pipeline(model_path: str):
    try:
        if not os.path.exists(model_path):
            return None
        return load_pipeline(model_path)
    except Exception as e:
        st.info(f"Skipping classic model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_stats_index(stats_xlsx: str):
    return RegionStatsIndex.from_excel(stats_xlsx)


@st.cache_resource(show_spinner=False)
def get_nn_artifacts():
    base_dir = os.environ.get("NN_MODEL_DIR", os.path.join(os.path.dirname(__file__), "nn_model"))
    try:
        artifacts = load_nn_artifacts(base_dir)
        return artifacts
    except Exception as e:
        st.info(f"NN artifacts not found/loaded from {base_dir}: {e}")
        return None


def render_map(df_pred: pd.DataFrame):
    if df_pred.empty:
        st.info("No rows to map.")
        return
    lat, lon = default_map_center(df_pred)
    # Build base map with layer control and satellite option
    m = folium.Map(location=[lat, lon], zoom_start=5, tiles=None)
    # Prefer Esri Satellite visible by default, OSM available via control
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri — Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri Satellite",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM", overlay=False, control=True, show=False).add_to(m)
    # Show cursor lat/lon and click lat/lon on the multi-point map
    try:
        MousePosition(position="topright", prefix="Lat/Lon:", separator=", ", num_digits=6).add_to(m)
    except Exception:
        pass
    try:
        folium.LatLngPopup().add_to(m)
    except Exception:
        pass

    lat_vals = []
    lon_vals = []
    for _, r in df_pred.iterrows():
        if pd.isna(r.get("LATITUDE")) or pd.isna(r.get("LONGITUDE")):
            continue
        try:
            lat_vals.append(float(r.get("LATITUDE")))
            lon_vals.append(float(r.get("LONGITUDE")))
        except Exception:
            pass
        # Build popup with only user-provided fields + predicted price
        user_fields = [
            ("Rooms", r.get("ROOMS", "")),
            ("Total area (m²)", r.get("TOTAL_AREA", "")),
            ("Floor/Total", f"{r.get('FLOOR','')}/{r.get('TOTAL_FLOORS','')}")
        ]
        extra_fields = [
            ("Furniture", r.get("FURNITURE", "")),
            ("Condition", r.get("CONDITION", "")),
            ("Ceiling (m)", r.get("CEILING", "")),
            ("Material", r.get("MATERIAL", "")),
            ("Year", r.get("YEAR", "")),
        ]
        lines = [f"<b>Predicted Price:</b> {r['pred_price_kzt']:,.0f} KZT"]
        # Add coordinates to popup for quick reference
        try:
            lines.append(f"<b>Coordinates:</b> {float(r['LATITUDE']):.6f}, {float(r['LONGITUDE']):.6f}")
        except Exception:
            pass
        lines += [f"<b>{k}:</b> {v}" for k, v in user_fields]
        lines += [f"<b>{k}:</b> {v}" for k, v in extra_fields]
        popup_html = folium.Popup(html="<br/>".join(lines), max_width=350)
        # Point-style icon (circle)
        folium.CircleMarker(
            location=[r["LATITUDE"], r["LONGITUDE"]],
            radius=6,
            color="#00E0A8",
            fill=True,
            fill_color="#00E0A8",
            fill_opacity=0.8,
            popup=popup_html,
            tooltip=f"{r['pred_price_kzt']:,.0f} KZT",
        ).add_to(m)
    # Add in-map search control
    if _FoliumGeocoder is not None:
        try:
            _FoliumGeocoder(collapsed=False, add_marker=True, position="topleft", placeholder="Search place").add_to(m)
        except Exception:
            pass
    if _Fullscreen is not None:
        try:
            _Fullscreen(position="topright").add_to(m)
        except Exception:
            pass
    folium.LayerControl(collapsed=True).add_to(m)
    # Fit bounds to all points with ~10 km buffer (in degrees)
    if lat_vals and lon_vals:
        import math
        min_lat, max_lat = min(lat_vals), max(lat_vals)
        min_lon, max_lon = min(lon_vals), max(lon_vals)
        mean_lat = sum(lat_vals) / len(lat_vals)
        buffer_lat = 10.0 / 111.0  # ~1 deg = 111 km
        cos_lat = max(0.1, math.cos(math.radians(abs(mean_lat))))
        buffer_lon = 10.0 / (111.0 * cos_lat)
        south = min_lat - buffer_lat
        north = max_lat + buffer_lat
        west = min_lon - buffer_lon
        east = max_lon + buffer_lon
        m.fit_bounds([[south, west], [north, east]])
    st_folium(m, height=650, use_container_width=True)


def geocode_nominatim(query: str, accept_language: str = "ru,kk,en", viewbox=None, bounded: bool = False):
    """Geocode address text to (lat, lon) using Nominatim.
    Returns (lat, lon) floats or (None, None) on failure.
    """
    try:
        if not query or not query.strip():
            return None, None
        base_url = "https://nominatim.openstreetmap.org"
        search_url = f"{base_url}/search"
        contact = os.environ.get("NOMINATIM_EMAIL", "")
        ua = "kz-real-estate-price-estimator/1.0 (+https://github.com/andprov/krisha.kz)"
        if contact:
            ua = f"{ua} {contact}"
        headers = {"User-Agent": ua}
        q = query.strip()
        # Try several variants to increase hit rate, biasing to Kazakhstan
        attempts = []
        attempts.append(q)
        if "," not in q and any(c.isdigit() for c in q):
            # If user typed just street+house, add Almaty as default city
            attempts.append(f"Алматы, {q}")
            attempts.append(f"Almaty, {q}")
        # Always try with country/city bias
        attempts.append(f"Kazakhstan, {q}")
        attempts.append(f"Казахстан, {q}")
        email_param = {"email": contact} if contact else {}

        # Free-form search attempts (single best)
        for qi in attempts:
            params = {
                "q": qi,
                "format": "jsonv2",
                "limit": 1,
                "accept-language": accept_language,
                "countrycodes": "kz",
                **email_param,
            }
            if viewbox:
                west, south, east, north = viewbox
                params["viewbox"] = f"{west},{north},{east},{south}"
                if bounded:
                    params["bounded"] = 1
            resp = requests.get(search_url, headers=headers, params=params, timeout=20)
            if resp.ok:
                data = resp.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    return lat, lon

        # Structured search fallback: parse 'City, Street House' or 'Street House, City'
        city = None
        street = None
        parts = [p.strip() for p in q.split(",") if p.strip()]
        if len(parts) >= 2:
            # Heuristic: first part city, second part street
            city, street = parts[0], ", ".join(parts[1:])
        else:
            # If no comma, try to use viewbox city context only with street
            street = q
        if street:
            struct_params = {
                "street": street,
                "format": "jsonv2",
                "limit": 1,
                "accept-language": accept_language,
                "countrycodes": "kz",
                **email_param,
            }
            if city:
                struct_params["city"] = city
            if viewbox:
                west, south, east, north = viewbox
                struct_params["viewbox"] = f"{west},{north},{east},{south}"
                if bounded:
                    struct_params["bounded"] = 1
            resp2 = requests.get(search_url, headers=headers, params=struct_params, timeout=20)
            if resp2.ok:
                data2 = resp2.json()
                if data2:
                    lat = float(data2[0]["lat"])
                    lon = float(data2[0]["lon"])
                    return lat, lon
        return None, None
    except Exception:
        return None, None


def _nominatim_headers() -> dict:
    """Respect Nominatim policy by including a contact in the User-Agent."""
    email = os.environ.get("NOMINATIM_EMAIL", "")
    try:
        if not email and hasattr(st, "secrets") and isinstance(st.secrets, dict) and "NOMINATIM_EMAIL" in st.secrets:
            email = st.secrets["NOMINATIM_EMAIL"]
    except Exception:
        pass
    ua_extra = f" (+https://github.com/andprov/krisha.kz); {email}" if email else " (+https://github.com/andprov/krisha.kz)"
    return {"User-Agent": f"kz-house-price-app/1.0{ua_extra}"}


# -------- Alternate Geocoding Providers (Mapbox, HERE, Photon) --------
def _get_secret(name: str, default: str = "") -> str:
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        if hasattr(st, "secrets") and isinstance(st.secrets, dict) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default


def _choose_geocoder_provider() -> str:
    """Pick provider in priority: MAPBOX → HERE → GOOGLE → OPENCAGE → LOCATIONIQ → YANDEX → GEOAPIFY → MAPTILER → PHOTON."""
    if _get_secret("MAPBOX_TOKEN"):
        return "mapbox"
    if _get_secret("HERE_API_KEY"):
        return "here"
    if _get_secret("GOOGLE_MAPS_API_KEY"):
        return "google"
    if _get_secret("OPENCAGE_API_KEY"):
        return "opencage"
    if _get_secret("LOCATIONIQ_TOKEN"):
        return "locationiq"
    if _get_secret("YANDEX_API_KEY"):
        return "yandex"
    if _get_secret("GEOAPIFY_API_KEY"):
        return "geoapify"
    if _get_secret("MAPTILER_API_KEY"):
        return "maptiler"
    return "photon"


@st.cache_data(show_spinner=False, ttl=60)
def _suggest_mapbox(query: str, limit: int = 7, lang: str = "ru"):
    try:
        token = _get_secret("MAPBOX_TOKEN")
        if not token:
            return []
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(query)}.json"
        params = {
            "access_token": token,
            "limit": limit,
            "language": lang,
            "country": "kz",
            "types": "address,place,locality,neighborhood,poi",
        }
        r = requests.get(url, params=params, timeout=15)
        if not r.ok:
            return []
        js = r.json() or {}
        out = []
        for feat in js.get("features", []):
            try:
                center = feat.get("center", [])
                if len(center) != 2:
                    continue
                lon, lat = float(center[0]), float(center[1])
                name = feat.get("place_name", f"{lat:.6f}, {lon:.6f}")
                out.append({"lat": lat, "lon": lon, "label": name, "source": "mapbox"})
            except Exception:
                continue
        return out
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _suggest_here(query: str, limit: int = 7, lang: str = "ru"):
    try:
        api_key = _get_secret("HERE_API_KEY")
        if not api_key:
            return []
        url = "https://geocode.search.hereapi.com/v1/geocode"
        lang_code = "ru-RU" if lang.startswith("ru") else "en-US"
        params = {
            "q": query,
            "in": "countryCode:KAZ",
            "lang": lang_code,
            "limit": limit,
            "apiKey": api_key,
        }
        r = requests.get(url, params=params, timeout=15)
        if not r.ok:
            return []
        js = r.json() or {}
        out = []
        for it in js.get("items", []):
            try:
                pos = it.get("position", {})
                lat, lon = float(pos.get("lat")), float(pos.get("lng"))
                title = it.get("title") or f"{lat:.6f}, {lon:.6f}"
                out.append({"lat": lat, "lon": lon, "label": title, "source": "here"})
            except Exception:
                continue
        return out
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _suggest_photon(query: str, limit: int = 7, lang: str = "ru"):
    try:
        url = "https://photon.komoot.io/api/"
        # Bias to Kazakhstan: bbox approx (minLon,minLat,maxLon,maxLat) and current pointer
        min_lon, min_lat, max_lon, max_lat = 46.0, 40.0, 88.0, 56.0
        at_lat = float(st.session_state.get("LATITUDE", 43.238))
        at_lon = float(st.session_state.get("LONGITUDE", 76.886))
        params = {
            "q": query,
            "limit": limit,
            "lang": lang,
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "lat": f"{at_lat:.6f}",
            "lon": f"{at_lon:.6f}",
        }
        r = requests.get(url, params=params, timeout=15)
        if not r.ok:
            return []
        js = r.json() or {}
        out = []
        for f in js.get("features", []):
            try:
                coords = f.get("geometry", {}).get("coordinates", [])
                if len(coords) != 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])
                props = f.get("properties", {})
                name = props.get("name")
                city = props.get("city")
                country = props.get("country")
                label = name or city or country or f"{lat:.6f}, {lon:.6f}"
                out.append({"lat": lat, "lon": lon, "label": label, "source": "photon"})
            except Exception:
                continue
        return out
    except Exception:
        return []


def suggest_candidates(query: str, limit: int = 7, lang: str = "ru"):
    prov = _choose_geocoder_provider()
    if prov == "mapbox":
        return _suggest_mapbox(query, limit=limit, lang=lang)
    if prov == "here":
        return _suggest_here(query, limit=limit, lang=lang)
    # default photon
    return _suggest_photon(query, limit=limit, lang=lang)


def geocode_candidates(address: str, limit: int = 10, debug: bool = False):
    """Return candidate dicts {lat, lon, label, source} using Mapbox or HERE if configured; Photon otherwise."""
    out = []
    if not address or not address.strip():
        return out

    def _get(url, params=None, headers=None, timeout=12, attempts=2):
        last_exc = None
        for _ in range(attempts):
            try:
                return requests.get(url, params=params, headers=headers, timeout=timeout)
            except Exception as e:
                last_exc = e
                time.sleep(0.5)
        if last_exc:
            raise last_exc

    q = address.strip()

    # Provider detection
    mapbox_token = _get_secret("MAPBOX_TOKEN")
    here_key = _get_secret("HERE_API_KEY")
    google_key = _get_secret("GOOGLE_MAPS_API_KEY")
    oc_key = _get_secret("OPENCAGE_API_KEY")
    liq_key = _get_secret("LOCATIONIQ_TOKEN")
    yandex_key = _get_secret("YANDEX_API_KEY")
    geoapify_key = _get_secret("GEOAPIFY_API_KEY")
    maptiler_key = _get_secret("MAPTILER_API_KEY")

    # 1) Mapbox Geocoding API (autocomplete)
    if mapbox_token:
        try:
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(q)}.json"
            params = {
                "access_token": mapbox_token,
                "autocomplete": "true",
                "limit": limit,
                "language": "ru,en,kk",
                "country": "kz",
                "types": "address,place,locality,neighborhood,poi",
            }
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            feats = js.get("features", [])
            for f in feats:
                try:
                    center = f.get("center") or []
                    if len(center) != 2:
                        continue
                    lon, lat = float(center[0]), float(center[1])
                    label = f.get("place_name") or f.get("text") or f.get("id") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "mapbox"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"Mapbox error: {e}")

    # 2) HERE Autosuggest (requires apiKey)
    if here_key:
        try:
            # Use current map position as context if available
            at_lat = float(st.session_state.get("LATITUDE", 43.238))
            at_lon = float(st.session_state.get("LONGITUDE", 76.886))
            url = "https://autosuggest.search.hereapi.com/v1/autosuggest"
            params = {
                "q": q,
                "at": f"{at_lat:.6f},{at_lon:.6f}",
                "apiKey": here_key,
                "limit": limit,
                "lang": "ru-RU,en-US,kk-KZ",
                "in": "countryCode:KAZ",
            }
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            items = js.get("items", [])
            for it in items:
                pos = it.get("position")
                if not pos:
                    continue
                try:
                    lat = float(pos.get("lat")); lon = float(pos.get("lng"))
                    label = it.get("title") or it.get("address", {}).get("label") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "here"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"HERE error: {e}")

    # 3) Google Maps Geocoding API
    if google_key:
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {"address": q, "key": google_key, "language": "ru"}
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            for res in js.get("results", [])[:limit]:
                try:
                    loc = res.get("geometry", {}).get("location", {})
                    lat, lon = float(loc.get("lat")), float(loc.get("lng"))
                    label = res.get("formatted_address") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "google"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"Google error: {e}")

    # 4) OpenCage
    if oc_key:
        try:
            url = "https://api.opencagedata.com/geocode/v1/json"
            params = {"q": q, "key": oc_key, "limit": limit, "language": "ru", "countrycode": "kz"}
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            for item in js.get("results", []):
                try:
                    g = item.get("geometry", {})
                    lat, lon = float(g.get("lat")), float(g.get("lng"))
                    label = item.get("formatted") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "opencage"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"OpenCage error: {e}")

    # 5) LocationIQ
    if liq_key:
        try:
            url = "https://eu1.locationiq.com/v1/search"
            params = {"q": q, "key": liq_key, "format": "json", "limit": limit, "accept-language": "ru", "countrycodes": "kz"}
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            for item in js[:limit]:
                try:
                    lat, lon = float(item.get("lat")), float(item.get("lon"))
                    label = item.get("display_name") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "locationiq"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"LocationIQ error: {e}")

    # 6) Yandex Geocoder
    if yandex_key:
        try:
            url = "https://geocode-maps.yandex.ru/1.x/"
            params = {"geocode": q, "apikey": yandex_key, "format": "json", "lang": "ru_RU", "results": str(limit)}
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            feats = js.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
            for f in feats:
                try:
                    obj = f.get("GeoObject", {})
                    pos = obj.get("Point", {}).get("pos", "")
                    if not pos:
                        continue
                    lon_str, lat_str = pos.split()
                    lat, lon = float(lat_str), float(lon_str)
                    name = obj.get("name")
                    desc = obj.get("description")
                    label = f"{name}, {desc}" if name and desc else name or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "yandex"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"Yandex error: {e}")

    # 7) Geoapify
    if geoapify_key:
        try:
            url = "https://api.geoapify.com/v1/geocode/search"
            params = {"text": q, "apiKey": geoapify_key, "limit": limit, "format": "json", "lang": "ru", "filter": "countrycode:kz"}
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            for feat in js.get("results", []):
                try:
                    lat, lon = float(feat.get("lat")), float(feat.get("lon"))
                    label = feat.get("formatted") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "geoapify"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"Geoapify error: {e}")

    # 8) MapTiler Geocoding
    if maptiler_key:
        try:
            url = f"https://api.maptiler.com/geocoding/{requests.utils.quote(q)}.json"
            params = {"key": maptiler_key, "limit": limit, "language": "ru", "country": "kz"}
            resp = _get(url, params=params, timeout=12, attempts=2)
            resp.raise_for_status()
            js = resp.json()
            for feat in js.get("features", []):
                try:
                    center = feat.get("center") or []
                    if len(center) != 2:
                        continue
                    lon, lat = float(center[0]), float(center[1])
                    label = feat.get("place_name") or feat.get("text") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "maptiler"})
                except Exception:
                    continue
            if out:
                return out
        except Exception as e:
            if debug:
                st.warning(f"MapTiler error: {e}")

    # 9) Photon (no key) as default (bias to Kazakhstan bbox and current pointer)
    try:
        url2 = "https://photon.komoot.io/api/"
        min_lon, min_lat, max_lon, max_lat = 46.0, 40.0, 88.0, 56.0
        at_lat = float(st.session_state.get("LATITUDE", 43.238))
        at_lon = float(st.session_state.get("LONGITUDE", 76.886))
        for lang_try in ("ru", "en"):
            params2 = {
                "q": q,
                "limit": limit,
                "lang": lang_try,
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                "lat": f"{at_lat:.6f}",
                "lon": f"{at_lon:.6f}",
            }
            resp2 = _get(url2, params=params2, timeout=12, attempts=2)
            resp2.raise_for_status()
            js2 = resp2.json()
            feats = js2.get("features", [])
            for f in feats:
                try:
                    coords = f["geometry"]["coordinates"]
                    lon, lat = float(coords[0]), float(coords[1])
                    props = f.get("properties", {})
                    label = props.get("name") or props.get("city") or props.get("country") or f"{lat}, {lon}"
                    out.append({"lat": lat, "lon": lon, "label": label, "source": "photon"})
                except Exception:
                    continue
            if out:
                break
    except Exception as e:
        if debug:
            st.warning(f"Photon error: {e}")
    return out


def parse_coords(text: str):
    """Parse 'lat, lon' or 'lat lon' into floats; return (lat, lon) or (None, None)."""
    try:
        if not text:
            return None, None
        s = text.strip().replace(";", ",").replace("|", ",")
        m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[ ,]+\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
        if not m:
            return None, None
        lat = float(m.group(1))
        lon = float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
        return None, None
    except Exception:
        return None, None


def predict_on_df(pipeline, stats_idx, df_in: pd.DataFrame, nn_artifacts=None) -> pd.DataFrame:
    df, missing = validate_inputs(df_in)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # enrich with regional socio-economic features if available
    if stats_idx is None:
        df_aug = df.copy()
        # add empty enrichment columns so UI remains consistent
        for c in ["segment_code", "region_name", "srednmes_zarplata", "chislennost_naseleniya_092025"]:
            if c not in df_aug.columns:
                df_aug[c] = ["", "", float("nan"), float("nan")][["segment_code","region_name","srednmes_zarplata","chislennost_naseleniya_092025"].index(c)]
    else:
        df_aug = attach_region_features(df, stats_idx)
    if nn_artifacts is not None:
        # Apply NN categorical codes and predict (returns real KZT)
        df_enc = nn_apply_cat_maps(df_aug, nn_artifacts.cat_maps)
        preds = nn_predict_kzt(nn_artifacts, df_enc)
    elif pipeline is not None:
        # Fall back to classical pipeline
        preds = predict_kzt(pipeline, df_aug)
    else:
        raise RuntimeError("No model available. Provide NN artifacts (NN_MODEL_DIR) or baseline model.joblib (MODEL_PATH).")
    out = df_aug.copy()
    out["pred_price_kzt"] = preds
    return out


model_path = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.joblib"))
# Prefer local stats; fall back to parent folder (102025)
_local_stats = os.path.join(os.path.dirname(__file__), "Stat_KZ092025.xlsx")
_parent_stats = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Stat_KZ092025.xlsx")
stats_path = os.environ.get("STATS_XLSX", _local_stats if os.path.exists(_local_stats) else _parent_stats)

# Preload NN artifacts to decide messaging and reuse later
nn_artifacts_cached = get_nn_artifacts()

if not os.path.exists(model_path):
    if nn_artifacts_cached is None:
        st.warning("No baseline model found. Either train NN (train_nn.py) and set NN_MODEL_DIR, or train baseline (train_model.py). Proceeding without baseline.")
if not os.path.exists(stats_path):
    st.warning("Stats Excel (Stat_KZ092025.xlsx) not found. Place it in app folder or parent 102025, or set STATS_XLSX env var.")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Single Prediction")
    # Initialize session defaults for coordinates
    if "LATITUDE" not in st.session_state:
        st.session_state["LATITUDE"] = 43.238
    if "LONGITUDE" not in st.session_state:
        st.session_state["LONGITUDE"] = 76.886

    # In-map search and picker
    with st.expander("Pick coordinates", expanded=True):
        st.caption("Use the search box on the map to find a place, then click on the map to set coordinates.")
        pick_lat = st.session_state.get("LATITUDE", 43.238)
        pick_lon = st.session_state.get("LONGITUDE", 76.886)
        # If a prediction exists, center map tighter on that point
        _pred_df = st.session_state.get("single_pred_df")
        if isinstance(_pred_df, pd.DataFrame) and not _pred_df.empty:
            try:
                _r = _pred_df.iloc[0]
                pick_lat = float(_r.get("LATITUDE", pick_lat))
                pick_lon = float(_r.get("LONGITUDE", pick_lon))
                _zoom = 14
            except Exception:
                _zoom = 12
        else:
            _zoom = 12
        mp = folium.Map(location=[pick_lat, pick_lon], zoom_start=_zoom, tiles=None)
        # Prefer Esri Satellite visible by default, OSM available via control
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri",
            name="Esri Satellite",
            overlay=False,
            control=True,
            show=True,
        ).add_to(mp)
        folium.TileLayer("OpenStreetMap", name="OSM", overlay=False, control=True, show=False).add_to(mp)
        # Show coordinates under cursor and on click popup
        MousePosition(position="topright", prefix="Lat/Lon:", separator=", ", num_digits=6).add_to(mp)
        # In-map geocoder/search control
        if _FoliumGeocoder is not None:
            try:
                _FoliumGeocoder(collapsed=False, add_marker=True, position="topleft", placeholder="Search place").add_to(mp)
            except Exception:
                pass
        if _Fullscreen is not None:
            try:
                _Fullscreen(position="topright").add_to(mp)
            except Exception:
                pass
        folium.LatLngPopup().add_to(mp)
        # If we have a prediction, show it as a styled marker with price
        single_pred = st.session_state.get("single_pred_df")
        if isinstance(single_pred, pd.DataFrame) and not single_pred.empty:
            try:
                r = single_pred.iloc[0]
                plat = float(r.get("LATITUDE", pick_lat))
                plon = float(r.get("LONGITUDE", pick_lon))
                price = r.get("pred_price_kzt")
                popup_lines = [f"<b>Predicted Price:</b> {price:,.0f} KZT"] if pd.notna(price) else []
                popup = folium.Popup("<br/>".join(popup_lines), max_width=300) if popup_lines else None
                folium.CircleMarker(
                    location=[plat, plon],
                    radius=7,
                    color="#00E0A8",
                    fill=True,
                    fill_color="#00E0A8",
                    fill_opacity=0.85,
                    popup=popup,
                    tooltip=f"{price:,.0f} KZT" if pd.notna(price) else "Prediction",
                ).add_to(mp)
            except Exception:
                folium.Marker(location=[pick_lat, pick_lon], tooltip="Current").add_to(mp)
        else:
            folium.Marker(location=[pick_lat, pick_lon], tooltip="Current").add_to(mp)
        folium.LayerControl(collapsed=True).add_to(mp)
        map_state = st_folium(
            mp,
            key="picker_map",
            height=550,
            use_container_width=True,
        )
        # Extract click
        if map_state and isinstance(map_state, dict):
            click = map_state.get("last_clicked")
            if click and "lat" in click and "lng" in click:
                st.session_state["LATITUDE"] = float(click["lat"])
                st.session_state["LONGITUDE"] = float(click["lng"])
                st.success(f"Selected: {st.session_state['LATITUDE']:.6f}, {st.session_state['LONGITUDE']:.6f}")

    with st.form("form_single"):
        # Compact 3-column layout
        r1c1, r1c2, r1c3 = st.columns(3)
        ROOMS = r1c1.number_input("Rooms", min_value=1, max_value=10, value=2)
        TOTAL_AREA = r1c2.number_input("Area (m²)", min_value=10.0, max_value=1000.0, value=60.0, step=1.0)
        YEAR = r1c3.number_input("Year", min_value=1950, max_value=2100, value=2015)

        r2c1, r2c2, r2c3 = st.columns(3)
        FLOOR = r2c1.number_input("Floor", min_value=1, max_value=100, value=5)
        TOTAL_FLOORS = r2c2.number_input("Total floors", min_value=1, max_value=100, value=9)
        CEILING = r2c3.number_input("Ceiling (m)", min_value=2.0, max_value=5.0, value=2.7, step=0.1)

        r3c1, r3c2, r3c3 = st.columns(3)
        MATERIAL = r3c1.selectbox("Material", ["Panel", "Brick", "Monolithic", "Mixed"], index=1)
        FURNITURE = r3c2.selectbox("Furniture", ["No", "Partial", "Full"], index=0)
        CONDITION = r3c3.selectbox("Condition", ["Needs renovation", "Good", "Excellent"], index=1)

        r4c1, r4c2 = st.columns(2)
        LATITUDE = r4c1.number_input("Latitude", value=float(st.session_state["LATITUDE"]), help="Decimal degrees")
        LONGITUDE = r4c2.number_input("Longitude", value=float(st.session_state["LONGITUDE"]), help="Decimal degrees")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            pipeline = get_pipeline(model_path)
            try:
                stats_idx = get_stats_index(stats_path)
            except Exception:
                stats_idx = None
            df_row = pd.DataFrame([
                {
                    "ROOMS": ROOMS,
                    "TOTAL_AREA": TOTAL_AREA,
                    "FLOOR": FLOOR,
                    "TOTAL_FLOORS": TOTAL_FLOORS,
                    "FURNITURE": FURNITURE,
                    "CONDITION": CONDITION,
                    "CEILING": CEILING,
                    "MATERIAL": MATERIAL,
                    "YEAR": YEAR,
                    "LATITUDE": LATITUDE,
                    "LONGITUDE": LONGITUDE,
                }
            ])
            out = predict_on_df(pipeline, stats_idx, df_row, nn_artifacts=nn_artifacts_cached)
            # Remove enrichment columns from predictions
            _drop_cols = [
                "segment_code",
                "region_name",
                "srednmes_zarplata",
                "chislennost_naseleniya_092025",
            ]
            out = out.drop(columns=[c for c in _drop_cols if c in out.columns])
            st.session_state["single_pred_df"] = out
            st.success(f"Predicted price: {out.loc[0,'pred_price_kzt']:,.0f} KZT")
        except Exception as e:
            st.error(str(e))

    # Persisted details (map above already shows result if available)
    if "single_pred_df" in st.session_state:
        out = st.session_state["single_pred_df"]
        with st.expander("Details", expanded=True):
            cols = [c for c in [
                "ROOMS","TOTAL_AREA","FLOOR","TOTAL_FLOORS","FURNITURE","CONDITION","CEILING","MATERIAL","YEAR","LATITUDE","LONGITUDE","pred_price_kzt"
            ] if c in out.columns]
            st.dataframe(out[cols])

with colB:
    st.subheader("Batch Upload (CSV/XLSX)")
    st.caption("Template columns: " + ", ".join(REQUIRED_FEATURES))
    # Top row: uploader + side-by-side download buttons
    up_col, dl_csv_col, dl_xlsx_col = st.columns([3, 1, 1])
    upload = up_col.file_uploader("Drag and drop file here", type=["csv", "xlsx"], help="CSV or Excel (.xlsx)")
    # Download buttons use session_state buffers when available
    b_csv = st.session_state.get("batch_csv_bytes")
    b_csv_name = st.session_state.get("batch_csv_name", "predictions.csv")
    b_xlsx = st.session_state.get("batch_xlsx_bytes")
    b_xlsx_name = st.session_state.get("batch_xlsx_name", "predictions.xlsx")
    dl_csv_col.download_button(
        label="Download predictions (CSV)",
        data=(b_csv or b""),
        file_name=b_csv_name,
        mime="text/csv",
        disabled=(b_csv is None),
        use_container_width=True,
        key="dl_batch_csv_toprow",
    )
    dl_xlsx_col.download_button(
        label="Download predictions (Excel)",
        data=(b_xlsx or b""),
        file_name=b_xlsx_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=(b_xlsx is None),
        use_container_width=True,
        key="dl_batch_xlsx_toprow",
    )
    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                df_in = pd.read_csv(upload)
            else:
                df_in = pd.read_excel(upload)
            pipeline = get_pipeline(model_path)
            try:
                stats_idx = get_stats_index(stats_path)
            except Exception:
                stats_idx = None
            out = predict_on_df(pipeline, stats_idx, df_in, nn_artifacts=nn_artifacts_cached)
            # Remove enrichment columns from predictions
            _drop_cols = [
                "segment_code",
                "region_name",
                "srednmes_zarplata",
                "chislennost_naseleniya_092025",
            ]
            out = out.drop(columns=[c for c in _drop_cols if c in out.columns])
            st.session_state["batch_pred_df"] = out
            st.success(f"Predicted {len(out)} rows.")
            cols = [c for c in [
                "ROOMS","TOTAL_AREA","FLOOR","TOTAL_FLOORS","FURNITURE","CONDITION","CEILING","MATERIAL","YEAR","LATITUDE","LONGITUDE","pred_price_kzt"
            ] if c in out.columns]
            st.dataframe(out[cols].head(50))
            # Prepare downloads in session_state for the top-row buttons
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.session_state["batch_csv_bytes"] = csv_bytes
            _ts = time.strftime("%Y%m%d_%H%M%S")
            st.session_state["batch_csv_name"] = f"houseprice_predictions_{_ts}.csv"
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                out.to_excel(writer, index=False, sheet_name="predictions")
            st.session_state["batch_xlsx_bytes"] = xlsx_buf.getvalue()
            st.session_state["batch_xlsx_name"] = f"houseprice_predictions_{_ts}.xlsx"
            st.subheader("Map")
            render_map(out)
            # Compact template download under the batch map
            with st.expander("Download batch template", expanded=False):
                st.caption("CSV/XLSX headers for batch upload")
                template_cols = [
                    "ROOMS","TOTAL_AREA","FLOOR","TOTAL_FLOORS","FURNITURE","CONDITION","CEILING","MATERIAL","YEAR","LATITUDE","LONGITUDE"
                ]
                template_df = pd.DataFrame(columns=template_cols)
                t1, t2 = st.columns(2)
                t1.download_button(
                    label="template.csv",
                    data=template_df.to_csv(index=False).encode("utf-8"),
                    file_name="template.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_template_csv_under_map",
                )
                xlsx_buf2 = io.BytesIO()
                with pd.ExcelWriter(xlsx_buf2, engine="openpyxl") as writer:
                    template_df.to_excel(writer, index=False, sheet_name="template")
                t2.download_button(
                    label="template.xlsx",
                    data=xlsx_buf2.getvalue(),
                    file_name="template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_template_xlsx_under_map",
                )
        except Exception as e:
            st.error(str(e))

# (Template download moved under the batch map)
