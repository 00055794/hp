import re
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    # A common desktop UA; randomized per session below
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}


@dataclass
class Listing:
    url: str
    listing_id: Optional[str] = None
    title: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    year: Optional[int] = None
    total_area: Optional[float] = None
    rooms: Optional[int] = None
    floor: Optional[int] = None
    total_floors: Optional[int] = None
    furniture: Optional[str] = None
    condition: Optional[str] = None
    ceiling: Optional[float] = None
    material: Optional[str] = None
    description: Optional[str] = None
    published_at: Optional[str] = None
    raw_attrs: Optional[Dict[str, str]] = None

    def to_row(self) -> Dict:
        d = asdict(self)
        # Standardize column names as requested
        d_out = {
            "URL": d.pop("url"),
            "ID": d.pop("listing_id"),
            "TITLE": d.pop("title"),
            "PRICE": d.pop("price"),
            "CURRENCY": d.pop("currency"),
            "ADDRESS": d.pop("address"),
            "CITY": d.pop("city"),
            "LATITUDE": d.pop("latitude"),
            "LONGITUDE": d.pop("longitude"),
            "YEAR": d.pop("year"),
            "TOTAL AREA": d.pop("total_area"),
            "ROOMS": d.pop("rooms"),
            "FLOOR": d.pop("floor"),
            "TOTAL_FLOORS": d.pop("total_floors"),
            "FURNITURE": d.pop("furniture"),
            "CONDITION": d.pop("condition"),
            "CEILING": d.pop("ceiling"),
            "MATERIAL": d.pop("material"),
            "DESCRIPTION": d.pop("description"),
            "PUBLISHED_AT": d.pop("published_at"),
            "RAW_ATTRS": json.dumps(d.pop("raw_attrs") or {}, ensure_ascii=False),
        }
        # Include any remaining keys (if any)
        for k, v in d.items():
            d_out[k] = v
        return d_out


class KrishaScraper:
    def __init__(self, base_url: str = "https://krisha.kz/", delay_range: Tuple[float, float] = (1.0, 2.5), session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip('/') + '/'
        self.delay_range = delay_range
        self.sess = session or requests.Session()
        # Randomize UA a little to reduce repetition
        ua_suffix = str(random.randint(1000, 9999))
        headers = DEFAULT_HEADERS.copy()
        headers["User-Agent"] += f" SafariPatch/{ua_suffix}"
        self.sess.headers.update(headers)

    def _sleep(self):
        time.sleep(random.uniform(*self.delay_range))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True,
           retry=retry_if_exception_type((requests.RequestException,)) )
    def _get(self, url: str, **kwargs) -> requests.Response:
        resp = self.sess.get(url, timeout=20, **kwargs)
        resp.raise_for_status()
        return resp

    def _with_page(self, url: str, page: int) -> str:
        # Add or replace page query param
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        qs["page"] = [str(page)]
        new_query = urlencode({k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in qs.items()}, doseq=True)
        return parsed._replace(query=new_query).geturl()

    def get_listing_urls(self, search_url: str, pages: int = 1, max_listings: Optional[int] = None) -> List[str]:
        found: List[str] = []
        for p in range(1, pages + 1):
            url = self._with_page(search_url, p)
            self._sleep()
            html = self._get(url).text
            soup = BeautifulSoup(html, 'lxml')
            # Krisha listing links typically include '/a/show/'
            links = set()
            for a in soup.select('a[href*="/a/show/"]'):
                href = a.get('href')
                if not href:
                    continue
                full = urljoin(self.base_url, href)
                links.add(full.split('?')[0])
            # Fallback: cards may have data-id
            for a in soup.select('a.a-card__title, a.a-card__image'):
                href = a.get('href')
                if href:
                    full = urljoin(self.base_url, href)
                    links.add(full.split('?')[0])
            page_links = sorted(links)
            found.extend([u for u in page_links if u not in found])
            if max_listings and len(found) >= max_listings:
                return found[:max_listings]
        return found

    def _extract_coords(self, html: str) -> Tuple[Optional[float], Optional[float]]:
        # Look for common JSON patterns containing lat/lon
        # Pattern 1: "lat": 43.256, "lng": 76.95
        m = re.search(r'"lat"\s*:\s*([\-\d\.]+)\s*[,}]', html, re.I)
        n = re.search(r'"lng"|"lon(gitude)?"\s*:\s*([\-\d\.]+)\s*[,}]', html, re.I)
        lat = float(m.group(1)) if m else None
        lon = None
        if n:
            # group 2 holds the numeric value per regex alternation
            try:
                lon = float(n.group(2))
            except Exception:
                lon = None
        # Pattern 2: latitude/longitude
        if lat is None or lon is None:
            m2 = re.search(r'"latitude"\s*:\s*([\-\d\.]+)', html, re.I)
            n2 = re.search(r'"longitude"\s*:\s*([\-\d\.]+)', html, re.I)
            if m2 and n2:
                try:
                    lat = float(m2.group(1))
                    lon = float(n2.group(1))
                except Exception:
                    pass
        # Sanity filter: Kazakhstan approx bounds
        if lat is not None and lon is not None:
            if not (40.0 <= lat <= 56.0 and 46.0 <= lon <= 90.0):
                return None, None
        return lat, lon

    def _extract_json_ld(self, soup: BeautifulSoup) -> Dict:
        data = {}
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                obj = json.loads(script.text.strip())
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            data.update(item)
                elif isinstance(obj, dict):
                    data.update(obj)
            except Exception:
                continue
        return data

    def _norm_text(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def _parse_table_attrs(self, soup: BeautifulSoup) -> Dict[str, str]:
        attrs: Dict[str, str] = {}
        # Try definition lists
        for dl in soup.select('dl, .params, .parameters'):
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            if dts and dds and len(dts) == len(dds):
                for dt, dd in zip(dts, dds):
                    k = self._norm_text(dt.get_text(" "))
                    v = self._norm_text(dd.get_text(" "))
                    if k and v:
                        attrs[k] = v
        # Try two-column table rows
        for row in soup.select('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) == 2:
                k = self._norm_text(cols[0].get_text(' '))
                v = self._norm_text(cols[1].get_text(' '))
                if k and v:
                    attrs[k] = v
        # Try generic key:value spans
        for item in soup.select('[class*="param"], [class*="feature"], li'):
            label = item.find(['span','div'], class_=re.compile(r'label|name|title', re.I))
            val = item.find(['span','div'], class_=re.compile(r'value|desc|text', re.I))
            if label and val:
                k = self._norm_text(label.get_text(' '))
                v = self._norm_text(val.get_text(' '))
                if k and v:
                    attrs[k] = v
        return attrs

    def _map_attrs(self, attrs: Dict[str, str]) -> Dict[str, Optional[str]]:
        mapping = {
            # Russian
            'Год постройки': 'YEAR', 'Год постройки, сдача': 'YEAR', 'Год': 'YEAR',
            'Общая площадь': 'TOTAL AREA', 'Площадь': 'TOTAL AREA', 'Площадь, м²': 'TOTAL AREA',
            'Количество комнат': 'ROOMS', 'Комнаты': 'ROOMS', 'Тип жилья': None,
            'Этаж': 'FLOOR', 'Этажность': 'TOTAL_FLOORS', 'Этажность дома': 'TOTAL_FLOORS',
            'Мебель': 'FURNITURE', 'Состояние': 'CONDITION', 'Ремонт': 'CONDITION',
            'Высота потолков': 'CEILING', 'Материал стен': 'MATERIAL', 'Город': 'CITY', 'Адрес': 'ADDRESS',
            # Kazakh
            'Салынған жылы': 'YEAR',
            'Жалпы ауданы': 'TOTAL AREA',
            'Бөлмелер': 'ROOMS', 'Бөлме саны': 'ROOMS',
            'Қабат': 'FLOOR', 'Қабаттылық': 'TOTAL_FLOORS',
            'Жиһаз': 'FURNITURE', 'Жағдайы': 'CONDITION',
            'Төбе биіктігі': 'CEILING', 'Қабырға материалы': 'MATERIAL', 'Қала': 'CITY', 'Мекенжайы': 'ADDRESS',
        }
        out: Dict[str, Optional[str]] = {}
        for k, v in attrs.items():
            key = None
            if k in mapping:
                key = mapping[k]
            else:
                # Try prefix matching for robustness
                for mkey, std in mapping.items():
                    if k.lower().startswith(mkey.lower()):
                        key = std
                        break
            if key is None:
                continue
            out[key] = v
        return out

    def _to_int(self, s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        m = re.search(r'\d+', s)
        try:
            return int(m.group(0)) if m else None
        except Exception:
            return None

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        if s is None:
            return None
        m = re.search(r'[\d\.\,]+', s)
        if not m:
            return None
        try:
            return float(m.group(0).replace(',', '.'))
        except Exception:
            return None

    def parse_listing(self, url: str) -> Listing:
        self._sleep()
        html = self._get(url).text
        soup = BeautifulSoup(html, 'lxml')
        json_ld = self._extract_json_ld(soup)

        title = self._norm_text(soup.find('h1').get_text(' ')) if soup.find('h1') else None
        # Price
        price_text = None
        currency = None
        price_el = soup.select_one('[itemprop="price"], .offer__price, .price')
        if price_el:
            price_text = self._norm_text(price_el.get_text(' '))
        # Currency from json-ld if possible
        if json_ld:
            currency = json_ld.get('priceCurrency') or currency
            if not price_text and json_ld.get('offers'):
                try:
                    price_text = str(json_ld['offers'].get('price'))
                except Exception:
                    pass
        # Address / City
        address = None
        city = None
        addr_el = soup.select_one('[itemprop="address"], .offer__location, .a-header__region')
        if addr_el:
            address = self._norm_text(addr_el.get_text(' '))
            # Heuristic city extraction: first comma-separated token often is city
            if address:
                city = address.split(',')[0].strip()
        if json_ld:
            adr = json_ld.get('address')
            if isinstance(adr, dict):
                address = adr.get('streetAddress') or address
                city = adr.get('addressLocality') or city
            elif isinstance(adr, str):
                address = adr or address

        lat, lon = self._extract_coords(html)

        # Collect attributes
        raw_attrs = self._parse_table_attrs(soup)
        mapped = self._map_attrs(raw_attrs)

        # Floor might be like "5 из 12"; parse both
        floor = None
        total_floors = None
        if 'FLOOR' in mapped:
            nums = re.findall(r'\d+', mapped['FLOOR'])
            if nums:
                floor = int(nums[0])
                if len(nums) > 1:
                    total_floors = int(nums[1])
        # If separate TOTAL_FLOORS provided
        if total_floors is None and 'TOTAL_FLOORS' in mapped:
            total_floors = self._to_int(mapped['TOTAL_FLOORS'])

        # Year
        year = self._to_int(mapped.get('YEAR')) if 'YEAR' in mapped else None
        # Area
        total_area = self._to_float(mapped.get('TOTAL AREA')) if 'TOTAL AREA' in mapped else None
        # Rooms
        rooms = self._to_int(mapped.get('ROOMS')) if 'ROOMS' in mapped else None
        # Ceiling height often like 2.7 м
        ceiling = self._to_float(mapped.get('CEILING')) if 'CEILING' in mapped else None

        # Listing id: try from URL path
        m = re.search(r'/a/show/(\d+)', url)
        listing_id = m.group(1) if m else None

        # Description
        desc_el = soup.select_one('[itemprop="description"], .offer__description, .description')
        description = self._norm_text(desc_el.get_text(' ')) if desc_el else None

        # Published date
        pub_el = soup.find(string=re.compile(r'Опубликовано|Обновлено|Жарияланды', re.I))
        published_at = None
        if pub_el:
            published_at = self._norm_text(pub_el)

        return Listing(
            url=url,
            listing_id=listing_id,
            title=title,
            price=price_text,
            currency=currency,
            address=address,
            city=city or mapped.get('CITY'),
            latitude=lat,
            longitude=lon,
            year=year,
            total_area=total_area,
            rooms=rooms,
            floor=floor,
            total_floors=total_floors,
            furniture=mapped.get('FURNITURE'),
            condition=mapped.get('CONDITION'),
            ceiling=ceiling,
            material=mapped.get('MATERIAL'),
            description=description,
            published_at=published_at,
            raw_attrs=raw_attrs or None,
        )

    def scrape(self, search_url: str, pages: int = 1, max_listings: Optional[int] = None) -> pd.DataFrame:
        urls = self.get_listing_urls(search_url, pages=pages, max_listings=max_listings)
        rows: List[Dict] = []
        for i, u in enumerate(urls, 1):
            try:
                item = self.parse_listing(u)
                rows.append(item.to_row())
            except Exception as e:
                rows.append({
                    "URL": u,
                    "ERROR": str(e)
                })
        df = pd.DataFrame(rows)
        # Ensure requested columns exist even if empty
        requested = [
            'YEAR', 'LONGITUDE', 'LATITUDE', 'TOTAL AREA', 'ROOMS', 'FLOOR', 'TOTAL_FLOORS',
            'FURNITURE', 'CONDITION', 'CEILING', 'MATERIAL', 'CITY'
        ]
        for col in requested:
            if col not in df.columns:
                df[col] = None
        # Reorder columns: requested first, then others
        front = ['URL', 'ID', 'TITLE', 'PRICE', 'CURRENCY', 'ADDRESS', 'CITY', 'LATITUDE', 'LONGITUDE', 'YEAR',
                 'TOTAL AREA', 'ROOMS', 'FLOOR', 'TOTAL_FLOORS', 'FURNITURE', 'CONDITION', 'CEILING', 'MATERIAL',
                 'DESCRIPTION', 'PUBLISHED_AT', 'RAW_ATTRS']
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]
        return df
