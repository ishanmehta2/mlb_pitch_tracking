from __future__ import annotations
import json
import subprocess
from pathlib import Path

JSON_PATH  = Path("data/mlb-youtube-segmented.json")
RAW_DIR    = Path("raw_videos")
DOWNLOADER = "yt-dlp"          # make sure yt-dlp is installed


RAW_DIR.mkdir(parents=True, exist_ok=True)

with JSON_PATH.open(encoding="utf-8") as f:
    dataset: dict[str, dict] = json.load(f)

unique: dict[str, str] = {}
for meta in dataset.values():
    url = meta.get("url", "")
    if "v=" not in url:
        continue
    vid = url.split("v=")[-1].split("&")[0]
    if vid.startswith("RHlE"):
        continue   
    unique[vid] = url

print(f"{len(unique)} unique YouTube videos listed in JSON.")

for vid, url in unique.items():
    target = RAW_DIR / f"{vid}.mp4"
    if target.exists():
        print(f"[✓] {target.name} already exists — skip.")
        continue
    

    cmd = [
        DOWNLOADER,
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mkv",
        "-o", str(target),
        url,
    ]
    print(f"[↓] Downloading {url}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[✗] Download failed for {url}: {e}")

print("Finished downloading all raw videos.")
