"""Replace specific cities in Sentinel-2 shard outputs, then rebuild canonical outputs.

Typical use case:
1. Re-run a subset of cities into a new shard, for example:
   data/processed/s2_shards/bps_2017_2025_metadata.parquet
   data/processed/s2_shards/bps_2017_2025_embeddings.npy
2. Remove old rows for those same cities from existing shards.
3. Merge the cleaned shard set into canonical outputs.

This avoids duplicate building-year keys when a city is reprocessed after
better coordinates or improved preprocessing become available.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SHARD_DIR = BASE_DIR / "data" / "processed" / "s2_shards"
DEFAULT_STAGE_DIR = BASE_DIR / "data" / "processed" / "s2_shards_replaced"
DEFAULT_META_OUT = BASE_DIR / "data" / "processed" / "building_year_s2_metadata.parquet"
DEFAULT_EMB_OUT = BASE_DIR / "data" / "processed" / "building_year_s2_embeddings.npy"


def _emb_path_for(meta_path: Path) -> Path:
    return meta_path.with_name(meta_path.name.replace("_metadata.parquet", "_embeddings.npy"))


def _copy_as_is(meta_path: Path, stage_dir: Path) -> None:
    emb_path = _emb_path_for(meta_path)
    shutil.copy2(meta_path, stage_dir / meta_path.name)
    shutil.copy2(emb_path, stage_dir / emb_path.name)


def _filter_existing_shard(meta_path: Path, stage_dir: Path, cities_to_replace: set[str]) -> int:
    emb_path = _emb_path_for(meta_path)
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    emb = np.load(emb_path)
    if len(meta) != emb.shape[0]:
        raise ValueError(
            f"Row mismatch for {meta_path.name}: metadata has {len(meta)} rows, embeddings have {emb.shape[0]}"
        )

    if "city" not in meta.columns:
        raise ValueError(f"{meta_path.name} does not contain a 'city' column")

    keep_mask = ~meta["city"].isin(cities_to_replace)
    removed = int((~keep_mask).sum())
    kept_meta = meta.loc[keep_mask].reset_index(drop=True)
    kept_emb = emb[keep_mask.to_numpy()]

    out_meta = stage_dir / meta_path.name
    out_emb = stage_dir / emb_path.name
    kept_meta.to_parquet(out_meta, index=False)
    with open(out_emb, "wb") as f:
        np.save(f, kept_emb)
    return removed


def _merge_stage_dir(stage_dir: Path, meta_out: Path, emb_out: Path) -> None:
    meta_paths = sorted(stage_dir.glob("*_metadata.parquet"))
    if not meta_paths:
        raise FileNotFoundError(f"No shard metadata files found in {stage_dir}")

    metas: list[pd.DataFrame] = []
    embs: list[np.ndarray] = []
    for meta_path in meta_paths:
        emb_path = _emb_path_for(meta_path)
        meta = pd.read_parquet(meta_path).reset_index(drop=True)
        emb = np.load(emb_path)
        if len(meta) != emb.shape[0]:
            raise ValueError(
                f"Row mismatch for {meta_path.name}: metadata has {len(meta)} rows, embeddings have {emb.shape[0]}"
            )
        shard_name = meta_path.name.replace("_metadata.parquet", "")
        meta = meta.copy()
        meta["shard"] = shard_name
        metas.append(meta)
        embs.append(emb)
        print(f"[merge] {shard_name}: rows={len(meta):,} emb_shape={emb.shape}")

    merged_meta = pd.concat(metas, ignore_index=True)
    merged_emb = np.concatenate(embs, axis=0)

    dup_cols = [c for c in ["building_id", "year", "city"] if c in merged_meta.columns]
    dup_mask = merged_meta.duplicated(subset=dup_cols, keep=False)
    if dup_mask.any():
        sample = merged_meta.loc[dup_mask, dup_cols].head(10).to_dict("records")
        raise ValueError(
            f"Duplicate building-year keys detected after replacement on columns {dup_cols}. "
            f"Sample duplicates: {sample}"
        )

    merged_meta = merged_meta.reset_index(drop=True)
    merged_meta["row_id"] = np.arange(len(merged_meta), dtype=np.int64)

    meta_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_meta = meta_out.with_suffix(".parquet.tmp")
    tmp_emb = emb_out.with_suffix(".npy.tmp")
    merged_meta.to_parquet(tmp_meta, index=False)
    with open(tmp_emb, "wb") as f:
        np.save(f, merged_emb)
    tmp_meta.replace(meta_out)
    tmp_emb.replace(emb_out)

    print(f"[merge] wrote metadata -> {meta_out}")
    print(f"[merge] wrote embeddings -> {emb_out}  shape={merged_emb.shape}")
    if "status" in merged_meta.columns:
        print("[merge] status breakdown:")
        print(merged_meta["status"].value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, required=True, help="Comma-separated city names to replace")
    ap.add_argument("--replacement_meta", type=str, required=True, help="Path to new replacement metadata parquet")
    ap.add_argument("--shard_dir", type=str, default=str(DEFAULT_SHARD_DIR))
    ap.add_argument("--stage_dir", type=str, default=str(DEFAULT_STAGE_DIR))
    ap.add_argument("--meta_out", type=str, default=str(DEFAULT_META_OUT))
    ap.add_argument("--emb_out", type=str, default=str(DEFAULT_EMB_OUT))
    args = ap.parse_args()

    cities_to_replace = {c.strip() for c in args.cities.split(",") if c.strip()}
    replacement_meta = Path(args.replacement_meta)
    replacement_emb = _emb_path_for(replacement_meta)
    shard_dir = Path(args.shard_dir)
    stage_dir = Path(args.stage_dir)
    meta_out = Path(args.meta_out)
    emb_out = Path(args.emb_out)

    if not replacement_meta.exists():
        raise FileNotFoundError(f"Missing replacement metadata: {replacement_meta}")
    if not replacement_emb.exists():
        raise FileNotFoundError(f"Missing replacement embedding file: {replacement_emb}")
    if not shard_dir.exists():
        raise FileNotFoundError(f"Missing shard directory: {shard_dir}")

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    removed_total = 0
    for meta_path in sorted(shard_dir.glob("*_metadata.parquet")):
        if meta_path.resolve() == replacement_meta.resolve():
            continue
        removed = _filter_existing_shard(meta_path, stage_dir, cities_to_replace)
        removed_total += removed
        if removed:
            print(f"[filter] {meta_path.name}: removed {removed:,} rows for cities={sorted(cities_to_replace)}")
        else:
            print(f"[filter] {meta_path.name}: no matching rows removed")

    # Copy replacement shard last so the stage dir contains the new city results.
    _copy_as_is(replacement_meta, stage_dir)
    print(f"[replace] added replacement shard: {replacement_meta.name}")
    print(f"[replace] total removed rows from old shards: {removed_total:,}")

    _merge_stage_dir(stage_dir, meta_out, emb_out)


if __name__ == "__main__":
    main()
