"""
Scrape UHC policy PDFs from the commercial medical & drug policies page.

Usage:
    python scripts/scrape_policies.py                  # Download all policies
    python scripts/scrape_policies.py --limit 25       # Download first 25 only (for dev)
    python scripts/scrape_policies.py --dry-run        # Just list PDFs, don't download
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INSURER = os.getenv("INSURER", "uhc")
CONFIG_PATH = PROJECT_ROOT / "config" / "insurers" / f"{DEFAULT_INSURER}.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def fetch_pdf_links(config: dict) -> list[dict]:
    """Fetch the listing page and extract all policy PDF links."""
    url = config["listing_url"]
    headers = {"User-Agent": config["user_agent"]}

    print(f"Fetching listing page: {url}")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_prefix = config["pdf_url_prefix"]
    base_url = config["base_url"]

    policies = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Only include policy PDFs (not bulletins, archives, or medical records docs)
        if not href.endswith(".pdf"):
            continue
        if pdf_prefix not in href:
            continue
        # Skip monthly bulletins and archive files
        if "bulletin" in href.lower() or "archive" in href.lower() or "medical-record" in href.lower():
            continue

        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        # Extract policy name from link text or URL
        name = link.get_text(strip=True)
        if not name:
            # Fallback: derive from filename
            name = href.split("/")[-1].replace(".pdf", "").replace("-", " ").title()

        policies.append({
            "name": name,
            "url": full_url,
            "filename": href.split("/")[-1],
        })

    print(f"Found {len(policies)} policy PDFs")
    return policies


def download_pdfs(policies: list[dict], config: dict, limit: int | None = None):
    """Download policy PDFs to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    delay = config["scrape_delay_seconds"]
    headers = {"User-Agent": config["user_agent"]}

    if limit:
        policies = policies[:limit]
        print(f"Limiting to first {limit} policies (dev mode)")

    manifest = []
    failed = []

    for policy in tqdm(policies, desc="Downloading PDFs"):
        filepath = OUTPUT_DIR / policy["filename"]

        # Skip if already downloaded
        if filepath.exists() and filepath.stat().st_size > 0:
            manifest.append({
                **policy,
                "local_path": str(filepath),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "status": "cached",
            })
            continue

        try:
            resp = requests.get(policy["url"], headers=headers, timeout=60)
            resp.raise_for_status()

            filepath.write_bytes(resp.content)
            manifest.append({
                **policy,
                "local_path": str(filepath),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "file_size_kb": round(len(resp.content) / 1024, 1),
                "status": "downloaded",
            })
        except Exception as e:
            print(f"\n  Failed: {policy['filename']} — {e}")
            failed.append({"policy": policy["name"], "error": str(e)})

        time.sleep(delay)

    # Save manifest
    manifest_data = {
        "insurer": config["name"],
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total_found": len(policies),
        "total_downloaded": len(manifest),
        "total_failed": len(failed),
        "policies": manifest,
        "failures": failed,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nDone! {len(manifest)} downloaded, {len(failed)} failed")
    print(f"Manifest saved to: {MANIFEST_PATH}")

    if failed:
        print("\nFailed downloads:")
        for f_item in failed:
            print(f"  - {f_item['policy']}: {f_item['error']}")


def main():
    parser = argparse.ArgumentParser(description="Scrape insurance policy PDFs")
    parser.add_argument("--insurer", default=DEFAULT_INSURER, help="Insurer config name (default: uhc)")
    parser.add_argument("--limit", type=int, help="Download only N policies (for dev)")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs without downloading")
    args = parser.parse_args()

    global CONFIG_PATH
    CONFIG_PATH = PROJECT_ROOT / "config" / "insurers" / f"{args.insurer}.yaml"
    config = load_config()
    policies = fetch_pdf_links(config)

    if args.dry_run:
        print("\nDry run — policies found:")
        for i, p in enumerate(policies, 1):
            print(f"  {i:3d}. {p['name']}")
        print(f"\nTotal: {len(policies)} policies")
        return

    download_pdfs(policies, config, limit=args.limit)


if __name__ == "__main__":
    main()
