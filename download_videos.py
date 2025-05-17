#!/usr/bin/env python3
"""
Download every unique YouTube video referenced in
data/mlb-youtube-segmented.json and save as MKV.

We iterate over the values of the top-level dict because the JSON has the form
{
   "E40GRXPSLG7N": { "url": "...", ... },
   "IXYXNMN46GIO": { "url": "...", ... },
   ...
}
"""

import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
JSON_PATH = Path("data/mlb-youtube-segmented.json")   # <-- adjust if needed
SAVE_DIR  = Path("downloads")                         # target folder
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Use the actively maintained yt-dlp if available; fall back to youtube-dl.
DOWNLOADER = "yt-dlp"  # change to "youtube-dl" if yt-dlp isn't installed

# ---------------------------------------------------------------------
# Load the annotation dict
# ---------------------------------------------------------------------
with JSON_PATH.open("r", encoding="utf-8") as f:
    dataset = json.load(f)        # a dict: clip_id -> metadata dict

print(f"Loaded {len(dataset)} clips.")

# ---------------------------------------------------------------------
# Collect *unique* video URLs so we don't redownload the same video
# many times (several clips can come from the same YT video).
# ---------------------------------------------------------------------
unique_urls = {}
for meta in dataset.values():     # meta is the inner dict
    url = meta.get("url")
    if not url:
        continue
    yt_id = url.split("v=")[-1].split("&")[0]
    if yt_id.startswith("RHlE"):
        continue   
    unique_urls[yt_id] = url      # keeps only one url per id

print(f"Need to download {len(unique_urls)} unique videos.")

# ---------------------------------------------------------------------
# Download loop
# ---------------------------------------------------------------------
for yt_id, url in unique_urls.items():
    out_path = SAVE_DIR / f"{yt_id}.mkv"

    if out_path.exists():
        print(f"[✓] {out_path.name} already exists — skipping.")
        continue

    cmd = [
        DOWNLOADER,
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mkv",
        "-o", str(out_path),
        url,
    ]

    print(f"[→] Downloading {url}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[✗] Download failed for {url}: {e}")
