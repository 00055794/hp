import argparse
from pathlib import Path
import pandas as pd

from krisha_parser import KrishaScraper


def main():
    parser = argparse.ArgumentParser(description="Scrape krisha.kz listings into CSV/Parquet")
    parser.add_argument("search_url", help="Krisha search URL, e.g. https://krisha.kz/prodazha/kvartiry/almaty/?das[flat.priv_dorm]=2")
    parser.add_argument("--pages", type=int, default=1, help="Number of search pages to scan")
    parser.add_argument("--max-listings", type=int, default=None, help="Stop after N listings")
    parser.add_argument("--out", type=str, default="krisha_listings.csv", help="Output path (.csv or .parquet)")
    args = parser.parse_args()

    scraper = KrishaScraper()
    df = scraper.scrape(args.search_url, pages=args.pages, max_listings=args.max_listings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
