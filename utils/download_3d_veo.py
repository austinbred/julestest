#!/usr/bin/env python3
import os
import sys
import re
import requests
from urllib.parse import urlparse

# Optional: set a realistic browser User-Agent header
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/113.0.0.0 Safari/537.36'
    )
}

# Try importing JS renderer (requests-html which bundles pyppeteer)
try:
    from requests_html import HTMLSession
    JS_RENDER_AVAILABLE = True
except ImportError as e:
    print(f"❌ JS fallback disabled: {e}")
    HTMLSession = None
    JS_RENDER_AVAILABLE = False

VIDEO_REGEX = re.compile(
    r"https?://c\.veocdn\.com/[0-9a-fA-F\-]+/panorama/[^\"']+?\.mp4"
)


def debug_search(html, label):
    print(f"\n🔍 Debug: searching for 'c.veocdn.com' in {label} (length {len(html)} chars)")
    if 'c.veocdn.com' in html:
        for m in VIDEO_REGEX.finditer(html):
            print(f"  • Candidate in {label}: {m.group()}")
    else:
        print(f"  • No 'c.veocdn.com' occurrences in {label}")
    print(f"  • Keyword 'panorama' {'present' if 'panorama' in html else 'not found'} in {label}")


def load_html(source):
    """
    Load HTML from a local .html file or via HTTP GET.
    """
    if os.path.isfile(source) and source.lower().endswith('.html'):
        print(f"🔧 Loading local HTML file: {source}")
        with open(source, 'r', encoding='utf-8') as f:
            html = f.read()
        debug_search(html, 'local HTML')
        return html
    resp = requests.get(source, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def download_stereo_video(match_source, cookie=None):
    # 1. Load raw HTML
    try:
        html = load_html(match_source)
    except Exception:
        headers = HEADERS.copy()
        if cookie:
            headers['Cookie'] = cookie
        resp = requests.get(match_source, headers=headers)
        resp.raise_for_status()
        html = resp.text

    # Write raw HTML to downloads for debugging
    downloads_dir = os.path.expanduser("~/Downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    raw_path = os.path.join(downloads_dir, "veo_match_page.html")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"🛠️  Raw HTML saved to {raw_path}")

    # Search raw
    debug_search(html, 'raw HTML')
    matches = VIDEO_REGEX.findall(html)

    # 2. JS-rendered fallback
    if not matches:
        if JS_RENDER_AVAILABLE:
            print("🔄 No match in raw HTML; attempting JS render via requests-html...")
            try:
                session = HTMLSession()
                r = session.get(match_source, headers=HEADERS)
                r.html.render(timeout=30)
                rendered = r.html.html
                render_path = os.path.join(downloads_dir, "veo_match_page_rendered.html")
                with open(render_path, 'w', encoding='utf-8') as f2:
                    f2.write(rendered)
                print(f"🛠️  Rendered HTML saved to {render_path}")
                debug_search(rendered, 'rendered HTML')
                matches = VIDEO_REGEX.findall(rendered)
            except Exception as e:
                print(f"❌ JS render failed: {e}")
        else:
            print("❌ JS fallback unavailable. Ensure dependencies are installed:")
            print("   pip3 install requests-html lxml_html_clean")
            print("   or pip3 install 'lxml[html_clean]'")

    # 3. No URL found
    if not matches:
        print("❌ No stereoscopic 3D URL found after searching HTML.")
        sys.exit(1)

    # 4. Download first URL
    video_url = matches[0]
    print(f"✅ Found video URL: {video_url}")
    filename = os.path.basename(urlparse(video_url).path)
    dest = os.path.join(downloads_dir, filename)
    print(f"📥 Downloading to {dest}...")
    with requests.get(video_url, stream=True, headers=HEADERS) as vr:
        vr.raise_for_status()
        with open(dest, 'wb') as outf:
            for chunk in vr.iter_content(chunk_size=8192):
                if chunk:
                    outf.write(chunk)
    print("✅ Download complete.")
    return dest


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 download_3d_veo.py <match_url_or_html_path> [cookie]")
        sys.exit(1)
    download_stereo_video(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
