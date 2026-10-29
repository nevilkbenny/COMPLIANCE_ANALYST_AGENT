"""
Download SEBI circular PDFs, starting from a saved SEBI circulars listing page.

Why this approach: SEBI's circulars listing page (sebi.gov.in/legal/circulars/)
is a JavaScript-rendered single-page app, so a plain scraper can't easily read
its table. But when you save the page in your browser ("Save As" -> Webpage,
HTML only), the browser captures whatever was rendered at that moment -
including the circular detail-page links. This script reads that saved file,
then visits each detail page (a normal server-rendered page) to find and
download the actual circular PDF.

Usage:
    pip install requests beautifulsoup4 --break-system-packages
    python download_sebi_circulars.py --html_file "SEBI___Circulars.html" --output_dir "SEBI CIRCULARS"

Re-runnable: keeps a manifest of already-downloaded circulars, so re-running
after saving a new/updated listing page will only fetch new ones.
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

CIRCULAR_LINK_PATTERN = re.compile(
    r"https://www\.sebi\.gov\.in/legal/circulars/[^\"'\s]+\.html"
)


def extract_circular_links(html_file: Path) -> list[str]:
    """Find all SEBI circular detail-page links in the saved HTML file."""
    text = html_file.read_text(encoding="utf-8", errors="ignore")
    links = sorted(set(CIRCULAR_LINK_PATTERN.findall(text)))
    return links


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len]


def find_pdf_url(detail_page_html: str) -> str | None:
    """
    Given the HTML of a circular detail page, find the actual PDF URL.

    SEBI embeds the circular PDF inside an <iframe> (not an <a> tag):
        <iframe src="../../../web/?file=https://www.sebi.gov.in/.../123.pdf">
    The original code only checked <a href>, so it always missed the PDF.
    Fix: check <iframe src> first, then fall back to <a href>.
    """
    soup = BeautifulSoup(detail_page_html, "html.parser")

    def _extract_pdf(attr_value: str):
        # Case A: wrapped ?file=<actual pdf url>
        if "file=" in attr_value:
            parsed = urlparse(attr_value)
            qs = parse_qs(parsed.query)
            if "file" in qs:
                candidate = qs["file"][0]
                if candidate.lower().endswith(".pdf"):
                    return candidate
        # Case B: direct .pdf link
        if attr_value.lower().endswith(".pdf"):
            if attr_value.startswith("http"):
                return attr_value
            return "https://www.sebi.gov.in" + attr_value
        return None

    # Primary: <iframe src="..."> -- this is how SEBI serves circular detail pages
    for tag in soup.find_all("iframe", src=True):
        result = _extract_pdf(tag["src"])
        if result:
            return result

    # Fallback: <a href="...">
    for tag in soup.find_all("a", href=True):
        result = _extract_pdf(tag["href"])
        if result:
            return result

    return None


def get_circular_title_and_date(detail_page_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(detail_page_html, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "untitled"

    date = ""
    date_tag = soup.find(string=re.compile(r"\b(19|20)\d{2}\b"))
    if date_tag:
        date = date_tag.strip()

    return title, date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_file", required=True,
                         help="Path to the saved SEBI circulars listing HTML page")
    parser.add_argument("--output_dir", default="SEBI CIRCULARS")
    parser.add_argument("--manifest", default="./sebi_download_manifest.json")
    parser.add_argument("--delay", type=float, default=1.5,
                         help="Seconds to wait between requests (politeness)")
    args = parser.parse_args()

    html_file = Path(args.html_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    circular_links = extract_circular_links(html_file)
    print(f"Found {len(circular_links)} circular detail links in {html_file.name}\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for link in circular_links:
        if manifest.get(link):
            print(f"  [skip] already downloaded: {link}")
            skipped += 1
            continue

        try:
            print(f"  [fetch] {link}")
            resp = requests.get(link, headers=HEADERS, timeout=20)
            resp.raise_for_status()

            pdf_url = find_pdf_url(resp.text)
            if not pdf_url:
                print(f"    WARNING: no PDF link found on this page, skipping")
                failed += 1
                continue

            title, date = get_circular_title_and_date(resp.text)

            time.sleep(args.delay)  # politeness delay before the PDF request too

            pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
            pdf_resp.raise_for_status()

            filename = f"{slugify(title)}.pdf"
            out_path = output_dir / filename

            # Avoid overwriting if a different circular slugifies to the same name
            counter = 1
            while out_path.exists():
                out_path = output_dir / f"{slugify(title)}-{counter}.pdf"
                counter += 1

            out_path.write_bytes(pdf_resp.content)
            print(f"    saved: {out_path.name}  ({date})")

            manifest[link] = {
                "title": title,
                "date": date,
                "pdf_url": pdf_url,
                "saved_as": out_path.name,
            }
            downloaded += 1

        except requests.RequestException as e:
            print(f"    ERROR fetching {link}: {e}")
            failed += 1

        time.sleep(args.delay)

    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. Downloaded: {downloaded}, Skipped (already had): {skipped}, Failed: {failed}")
    print(f"PDFs saved to: {output_dir}")


if __name__ == "__main__":
    main()