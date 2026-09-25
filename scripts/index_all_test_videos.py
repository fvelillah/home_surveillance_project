#!/usr/bin/env python3
"""
Helper script to index all video clips in data/tests/ into Twelve Labs Marengo Index.

Ensures all test clips (e.g. test1.mp4 to test5.mp4) exist in 'dahua_surveillance_marengo'.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config
from backend import twelvelabs_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_all_test_videos")


def main() -> int:
    if not twelvelabs_client.is_enabled():
        print("TWELVE_LABS_API_KEY is not configured. Cannot index live clips.")
        return 1

    client = twelvelabs_client.get_client()
    index_id = twelvelabs_client.get_marengo_index_id()
    print(f"\n========================================================")
    print(f"  Indexing Videos into Twelve Labs Marengo Index")
    print(f"  Index Name: {config.marengo_index_name}")
    print(f"  Index ID:   {index_id}")
    print(f"========================================================\n")

    test_dir = PROJECT_ROOT / "data" / "tests"
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted([f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts])

    if not video_files:
        print(f"No video files found in '{test_dir}'")
        return 1

    print(f"Found {len(video_files)} video file(s) in {test_dir}:\n")
    for v in video_files:
        print(f"  - {v.name} ({v.stat().st_size / (1024*1024):.2f} MB)")

    print("\nProcessing uploads & indexing...")
    for idx, vid_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing '{vid_path.name}'...")
        try:
            res = twelvelabs_client.upload_video(vid_path, index_type="both")
            m_vid = res.get("marengo_video_id")
            p_vid = res.get("pegasus_video_id")
            print(f"  ✓ Indexed Marengo Video ID: {m_vid}")
            print(f"  ✓ Pegasus Asset ID:        {p_vid}")
        except Exception as exc:
            print(f"  ✗ Error indexing '{vid_path.name}': {exc}")

    print("\n========================================================")
    print("Verifying Indexed Assets in Marengo Index:")
    try:
        idx_assets_service = getattr(client.indexes, "indexed_assets", None)
        if idx_assets_service:
            indexed_items = list(idx_assets_service.list(index_id=index_id))
            print(f"Total Indexed Assets in index: {len(indexed_items)}")
            for item in indexed_items:
                print(f"  • ID: {getattr(item, 'id', None)} | Asset ID: {getattr(item, 'asset_id', None)} | Status: {getattr(item, 'status', None)}")
    except Exception as exc:
        print(f"Verification query failed: {exc}")

    print("\nDone!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
