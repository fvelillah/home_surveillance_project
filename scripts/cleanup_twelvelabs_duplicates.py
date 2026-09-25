#!/usr/bin/env python3
"""
Twelve Labs Asset Deduplication & Cleanup Tool.

Scans all uploaded assets in the Twelve Labs account, groups them by filename,
retains exactly one canonical ready asset per unique clip (ensuring it is indexed in Marengo),
and removes redundant duplicate assets.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config
from backend import twelvelabs_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup_twelvelabs_duplicates")


def main() -> int:
    if not twelvelabs_client.is_enabled():
        print("TWELVE_LABS_API_KEY is not configured.")
        return 1

    client = twelvelabs_client.get_client()
    marengo_index_id = twelvelabs_client.get_marengo_index_id()
    print("=" * 70)
    print("  TWELVE LABS ASSET DEDUPLICATION & CLEANUP TOOL")
    print(f"  Marengo Index ID: {marengo_index_id}")
    print("=" * 70)

    # 1. Fetch all indexed assets in Marengo index
    indexed_asset_ids = set()
    indexed_map = {}
    try:
        idx_assets_service = getattr(client.indexes, "indexed_assets", None)
        if idx_assets_service:
            for item in idx_assets_service.list(index_id=marengo_index_id):
                a_id = getattr(item, "asset_id", None) or getattr(item, "id", None)
                if a_id:
                    indexed_asset_ids.add(str(a_id))
                    indexed_map[str(a_id)] = str(getattr(item, "id", a_id))
    except Exception as exc:
        logger.warning("Could not list indexed assets: %s", exc)

    print(f"\nFound {len(indexed_asset_ids)} indexed asset(s) in Marengo index {marengo_index_id}.")

    # 2. Fetch all uploaded assets
    print("Listing all uploaded assets in Twelve Labs account...")
    all_assets = []
    try:
        pager = client.assets.list(page_limit=50)
        for a in pager:
            all_assets.append(a)
    except Exception as exc:
        print(f"Failed to list assets: {exc}")
        return 1

    print(f"Total uploaded assets in account: {len(all_assets)}\n")

    # 3. Group assets by filename or canonical name
    by_filename = defaultdict(list)
    for a in all_assets:
        fname = getattr(a, "filename", None) or "untitled"
        # Extract base name if path-like
        fname = Path(fname).name
        by_filename[fname].append(a)

    cache_data = {}
    to_delete = []

    print("Asset Inventory by File:")
    print("-" * 70)
    for fname, assets in sorted(by_filename.items()):
        print(f"📁 File: '{fname}' ({len(assets)} copy/copies)")
        # Sort: prefer indexed, then ready, then newest
        def asset_priority(asset):
            aid = str(getattr(asset, "id", ""))
            is_indexed = 1 if aid in indexed_asset_ids else 0
            is_ready = 1 if getattr(asset, "status", None) == "ready" else 0
            return (is_indexed, is_ready, str(getattr(asset, "created_at", "")))

        sorted_assets = sorted(assets, key=asset_priority, reverse=True)
        canonical = sorted_assets[0]
        canonical_id = str(getattr(canonical, "id", ""))
        status = getattr(canonical, "status", "unknown")

        print(f"   ✓ Keeping Canonical Asset: id={canonical_id} (status={status})")

        # Ensure canonical is indexed in Marengo if ready
        marengo_id = indexed_map.get(canonical_id, canonical_id)
        if status == "ready" and canonical_id not in indexed_asset_ids:
            try:
                idx_service = getattr(client.indexes, "indexed_assets", None)
                if idx_service:
                    print(f"   → Indexing canonical asset {canonical_id} into Marengo...")
                    idx_res = idx_service.create(index_id=marengo_index_id, asset_id=canonical_id)
                    marengo_id = str(getattr(idx_res, "id", canonical_id) or canonical_id)
                    indexed_asset_ids.add(canonical_id)
                    indexed_map[canonical_id] = marengo_id
            except Exception as exc:
                print(f"   ✗ Indexing error: {exc}")

        # Store in cache
        cache_data[fname] = {
            "asset_id": canonical_id,
            "marengo_video_id": marengo_id,
            "pegasus_video_id": canonical_id,
            "filename": fname,
            "status": status,
        }

        # Duplicates to delete
        duplicates = sorted_assets[1:]
        for dup in duplicates:
            dup_id = str(getattr(dup, "id", ""))
            to_delete.append((dup_id, fname))
            print(f"   ✗ Redundant Duplicate:   id={dup_id} (status={getattr(dup, 'status', 'unknown')})")
        print()

    print("-" * 70)
    print(f"Summary: {len(cache_data)} unique file(s) retained, {len(to_delete)} duplicate(s) to remove.")

    # 4. Delete duplicate assets
    if to_delete:
        print(f"\nDeleting {len(to_delete)} duplicate asset(s)...")
        deleted_count = 0
        for dup_id, fname in to_delete:
            try:
                client.assets.delete(asset_id=dup_id, force=True)
                deleted_count += 1
                print(f"  ✓ Deleted duplicate asset {dup_id} ({fname})")
            except Exception as exc:
                print(f"  ✗ Failed to delete asset {dup_id}: {exc}")
        print(f"\nSuccessfully removed {deleted_count} duplicate asset(s).")
    else:
        print("\nNo duplicates to remove.")

    # 5. Write local cache file
    cache_path = PROJECT_ROOT / "data" / "storage" / "twelvelabs_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    print(f"✓ Saved canonical asset cache to '{cache_path}'")
    print("\nCleanup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
