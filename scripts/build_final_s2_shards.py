#!/usr/bin/env python
"""Build a clean final S2 shard set from selected source shards.

This script:
1. Picks the intended best shard version for each city/year slice.
2. Filters out replaced cities from broader shards.
3. Deduplicates repeated building-year rows inside selected shards.
4. Writes a clean shard folder plus a manifest for auditing.

It does NOT overwrite the canonical merged S2 files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SHARD_DIR = BASE_DIR / "data" / "processed" / "s2_shards"
RETRY_WAVE1_DIR = BASE_DIR / "data" / "processed" / "s2_retry_wave1"
RETRY_TMR_DIR = BASE_DIR / "data" / "processed" / "s2_retry_wave_tmr"
AU_REFRESH_DIR = BASE_DIR / "data" / "processed" / "s2_au_geo_refresh"
PHILLY_REFRESH_DIR = BASE_DIR / "data" / "processed" / "s2_philadelphia_geo_refresh"
FINAL_DIR = BASE_DIR / "data" / "processed" / "s2_shards_final_v2"

SOURCE_DIRS = {
    "s2_shards": SHARD_DIR,
    "s2_retry_wave1": RETRY_WAVE1_DIR,
    "s2_retry_wave_tmr": RETRY_TMR_DIR,
    "s2_au_geo_refresh": AU_REFRESH_DIR,
    "s2_philadelphia_geo_refresh": PHILLY_REFRESH_DIR,
}


@dataclass(frozen=True)
class ShardSpec:
    output_name: str
    source_name: str
    source_dir: str = "s2_shards"
    include_cities: tuple[str, ...] | None = None
    exclude_cities: tuple[str, ...] | None = None


SPECS = [
    ShardSpec("bps_2017_2025_final", "bps_2017_2025_dedup"),
    ShardSpec("nyc_2017_2018_final", "nyc_2017_2018_wave1", source_dir="s2_retry_wave1"),
    ShardSpec("nyc_2019_2020_final", "nyc_2019_2020_wave1", source_dir="s2_retry_wave1"),
    ShardSpec("nyc_2021_2022_final", "nyc_2021_2022_wave1", source_dir="s2_retry_wave1"),
    ShardSpec("nyc_2023_2025_final", "nyc_2023_2025_wave1", source_dir="s2_retry_wave1"),
    ShardSpec("la_only_final", "la_only_wave1", source_dir="s2_retry_wave1"),
    ShardSpec(
        "pcs_2017_2025_final",
        "pcs_2017_2025_wave1",
        source_dir="s2_retry_wave1",
        include_cities=("chicago", "seattle"),
    ),
    ShardSpec(
        "philadelphia_final",
        "philadelphia_wave_building_patch_seed",
        source_dir="s2_philadelphia_geo_refresh",
        include_cities=("philadelphia",),
    ),
    ShardSpec("sf_only_final", "chicago_boston_sf", include_cities=("sf",)),
    ShardSpec("dc_only_final", "dc_only_tmr", source_dir="s2_retry_wave_tmr", include_cities=("dc",)),
    ShardSpec(
        "sydney_final",
        "au_wave_building_patch_seed",
        source_dir="s2_au_geo_refresh",
        include_cities=("sydney",),
    ),
    ShardSpec(
        "rest_world_final",
        "au_wave_building_patch_seed",
        source_dir="s2_au_geo_refresh",
        exclude_cities=("sydney",),
    ),
    ShardSpec(
        "denver_only_final",
        "rest_world_tmr",
        source_dir="s2_retry_wave_tmr",
        include_cities=("denver",),
    ),
]


def _paths_for(stem: str, source_dir: str) -> tuple[Path, Path]:
    base_dir = SOURCE_DIRS[source_dir]
    meta = base_dir / f"{stem}_metadata.parquet"
    emb = base_dir / f"{stem}_embeddings.npy"
    return meta, emb


def _score_status(status: str) -> int:
    if status == "ok":
        return 4
    if status == "no_images":
        return 3
    if "Too Many Requests" in status:
        return 2
    if "internal error" in status.lower():
        return 1
    return 0


def _score_img_source(img_source: str) -> int:
    if img_source == "building_patch":
        return 1
    return 0


def _load_and_filter(spec: ShardSpec) -> tuple[pd.DataFrame, np.ndarray]:
    meta_path, emb_path = _paths_for(spec.source_name, spec.source_dir)
    if not meta_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Missing source shard for {spec.source_name}")

    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    emb = np.load(emb_path)
    if len(meta) != emb.shape[0]:
        raise ValueError(
            f"Row mismatch for {spec.source_name}: meta={len(meta)} emb={emb.shape[0]}"
        )

    keep = pd.Series(True, index=meta.index)
    if spec.include_cities is not None:
        keep &= meta["city"].isin(spec.include_cities)
    if spec.exclude_cities is not None:
        keep &= ~meta["city"].isin(spec.exclude_cities)

    meta = meta.loc[keep].reset_index(drop=True)
    emb = emb[keep.to_numpy()]
    return meta, emb


def _dedup(meta: pd.DataFrame, emb: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, int]:
    if len(meta) == 0:
        return meta, emb, 0

    work = meta.copy()
    work["_status_rank"] = work["status"].map(_score_status)
    work["_img_rank"] = work["img_source"].map(_score_img_source)
    work["_n_images_rank"] = work["n_images"].fillna(-1)
    work["_orig_idx"] = np.arange(len(work), dtype=np.int64)

    work = work.sort_values(
        by=[
            "building_id",
            "year",
            "city",
            "_status_rank",
            "_img_rank",
            "_n_images_rank",
            "_orig_idx",
        ],
        ascending=[True, True, True, False, False, False, True],
        kind="mergesort",
    )
    keep = ~work.duplicated(subset=["building_id", "year", "city"], keep="first")
    removed = int((~keep).sum())

    kept = work.loc[keep].copy()
    emb_kept = emb[kept["_orig_idx"].to_numpy()]

    kept = kept.drop(columns=["_status_rank", "_img_rank", "_n_images_rank", "_orig_idx"])
    kept = kept.reset_index(drop=True)
    kept["row_id"] = np.arange(len(kept), dtype=np.int64)
    return kept, emb_kept, removed


def _write_shard(final_dir: Path, stem: str, meta: pd.DataFrame, emb: np.ndarray) -> None:
    meta_path = final_dir / f"{stem}_metadata.parquet"
    emb_path = final_dir / f"{stem}_embeddings.npy"
    meta.to_parquet(meta_path, index=False)
    with open(emb_path, "wb") as f:
        np.save(f, emb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir", type=str, default=str(FINAL_DIR))
    args = ap.parse_args()

    final_dir = Path(args.final_dir)
    manifest_path = final_dir / "manifest.csv"

    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for spec in SPECS:
        meta, emb = _load_and_filter(spec)
        before_rows = len(meta)
        deduped_meta, deduped_emb, removed = _dedup(meta, emb)
        _write_shard(final_dir, spec.output_name, deduped_meta, deduped_emb)

        row = {
            "output_name": spec.output_name,
            "source_name": spec.source_name,
            "source_dir": spec.source_dir,
            "before_rows": before_rows,
            "after_rows": len(deduped_meta),
            "removed_duplicates": removed,
            "ok_rows": int((deduped_meta["status"] == "ok").sum()),
            "building_patch_ok_rows": int(
                ((deduped_meta["status"] == "ok") & (deduped_meta["img_source"] == "building_patch")).sum()
            ),
            "city_fallback_ok_rows": int(
                ((deduped_meta["status"] == "ok") & (deduped_meta["img_source"] == "city_fallback")).sum()
            ),
            "cities": ",".join(sorted(deduped_meta["city"].dropna().unique())),
        }
        manifest_rows.append(row)
        print(
            f"{spec.output_name}: before={before_rows:,} after={len(deduped_meta):,} "
            f"removed_dup={removed:,} ok={row['ok_rows']:,}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
