#!/usr/bin/env python
"""Archive intermediate Sentinel-2 shard artifacts that are no longer canonical sources.

This script is conservative:
1. Reads `data/processed/s2_shards_final/manifest.csv` to determine live source shards.
2. Moves only known intermediate shard files into an archive directory.
3. Leaves canonical outputs, final shard sources, and final shard products untouched.

The move is reversible because files are archived rather than deleted.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SHARD_DIR = PROCESSED_DIR / "s2_shards"
FINAL_DIR = PROCESSED_DIR / "s2_shards_final"
MANIFEST_PATH = FINAL_DIR / "manifest.csv"
DEFAULT_ARCHIVE_DIR = PROCESSED_DIR / "s2_shards_archive_intermediate"


def _artifact_pair(stem: str, base_dir: Path) -> list[Path]:
    return [
        base_dir / f"{stem}_metadata.parquet",
        base_dir / f"{stem}_embeddings.npy",
    ]


def _bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def _load_live_sources() -> set[str]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    return set(manifest["source_name"].astype(str))


def _candidate_stems() -> list[str]:
    return [
        # NYC original and retry chains; geo_refresh is live and should be preserved.
        "nyc_2017_2018",
        "nyc_2017_2018_retry",
        "nyc_2017_2018_retry2",
        "nyc_2019_2020",
        "nyc_2019_2020_retry",
        "nyc_2019_2020_retry_retry2",
        "nyc_2021_2022",
        "nyc_2021_2022_retry",
        "nyc_2021_2022_retry_retry2",
        "nyc_2023_2025",
        "nyc_2023_2025_retry",
        "nyc_2023_2025_retry_retry2",
        # LA earlier stages; retry_retry2 is the live source.
        "la_only",
        "la_only_retry",
        # PCS earlier stage; retry2 is the live source.
        "pcs_2017_2025",
        # BPS earlier stage; dedup is the live source.
        "bps_2017_2025",
    ]


def _candidate_extra_files() -> list[Path]:
    return [
        PROCESSED_DIR / "nyc2020_s2_metadata.parquet",
        PROCESSED_DIR / "nyc2020_s2_embeddings.npy",
    ]


def _candidate_extra_dirs() -> list[Path]:
    return [
        PROCESSED_DIR / "s2_shards_replaced",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive_dir",
        type=str,
        default=str(DEFAULT_ARCHIVE_DIR),
        help="Destination directory for archived intermediate artifacts.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be archived without moving anything.",
    )
    args = ap.parse_args()

    archive_dir = Path(args.archive_dir)
    live_sources = _load_live_sources()

    shard_targets: list[Path] = []
    for stem in _candidate_stems():
        if stem in live_sources:
            continue
        shard_targets.extend([p for p in _artifact_pair(stem, SHARD_DIR) if p.exists()])

    extra_files = [p for p in _candidate_extra_files() if p.exists()]
    extra_dirs = [p for p in _candidate_extra_dirs() if p.exists()]

    all_targets = shard_targets + extra_files + extra_dirs
    if not all_targets:
        print("No intermediate S2 artifacts found to archive.")
        return

    total_bytes = 0
    print("Will archive the following artifacts:")
    for path in all_targets:
        size = 0
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        total_bytes += size
        print(f"  {path.relative_to(BASE_DIR)}  ({_bytes_to_gb(size):.2f} GB)")

    print(f"\nTotal to archive: {_bytes_to_gb(total_bytes):.2f} GB")
    if args.dry_run:
        print("Dry run only; no files moved.")
        return

    archive_dir.mkdir(parents=True, exist_ok=True)

    for path in all_targets:
        dest = archive_dir / path.relative_to(PROCESSED_DIR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))

    print(f"\nArchived intermediate S2 artifacts -> {archive_dir}")


if __name__ == "__main__":
    main()
