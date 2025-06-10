from __future__ import annotations
import json, multiprocessing as mp, subprocess
from pathlib import Path
from collections import defaultdict

JSON_PATH   = Path("data/mlb-youtube-segmented.json")
RAW_DIR     = Path("raw_videos")
CLIP_DIR    = Path("clips")
FFMPEG      = "ffmpeg"
RAW_SUFFIX  = ".mp4"        
NUM_WORKERS = 8

INDEX: dict[str, list] | None = None  


def _init_pool(shared_index: dict[str, list]) -> None:
    global INDEX
    INDEX = shared_index


def _worker(raw_path: Path) -> None:
    assert INDEX is not None, "Pool initializer did not set INDEX"
    vid = raw_path.stem
    jobs = INDEX.get(vid)
    if not jobs:
        # No matching clips for this raw file
        return

    for clip_id, start, duration in jobs:
        out = CLIP_DIR / f"{clip_id}.mp4"
        if out.exists():
            continue

        cmd = [
            FFMPEG,
            "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i",  str(raw_path),
            "-t",  f"{duration:.3f}",
            "-c:v", "copy", "-an",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[✗] ffmpeg failed for {clip_id}: {e}")


def build_index(dataset: dict[str, dict]) -> dict[str, list]:
    """video-id → list[(clip_id, start, duration)]."""
    idx: dict[str, list] = defaultdict(list)
    for cid, m in dataset.items():
        vid  = m["url"].split("v=")[-1].split("&")[0]
        start     = float(m["start"])
        duration  = float(m["end"]) - start
        idx[vid].append((cid, start, duration))
    return idx


def main() -> None:
    print("Loading annotations & building index...")
    RAW_DIR.mkdir(exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open(encoding="utf-8") as f:
        dataset = json.load(f)
    index = build_index(dataset)

    raw_files = [p for p in RAW_DIR.iterdir() if p.suffix.lower() == RAW_SUFFIX]
    print(f"Found {len(raw_files)} {RAW_SUFFIX} files in {RAW_DIR}")

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_pool,
        initargs=(index,),
    ) as pool:
        pool.map(_worker, raw_files)

    print("All requested clips are in", CLIP_DIR.resolve())


if __name__ == "__main__":
    main()
