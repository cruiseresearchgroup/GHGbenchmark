"""Merge per-shard Sentinel-2 embedding outputs into the canonical files.

Expected shard files:
  data/processed/s2_shards/*_metadata.parquet
  data/processed/s2_shards/*_embeddings.npy

Output:
  data/processed/building_year_s2_metadata.parquet
  data/processed/building_year_s2_embeddings.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SHARD_DIR = BASE_DIR / "data" / "processed" / "s2_shards"
DEFAULT_META_OUT = BASE_DIR / "data" / "processed" / "building_year_s2_metadata.parquet"
DEFAULT_EMB_OUT = BASE_DIR / "data" / "processed" / "building_year_s2_embeddings.npy"


def load_shard(meta_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    emb_path = meta_path.with_name(meta_path.name.replace("_metadata.parquet", "_embeddings.npy"))
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing matching embedding file for {meta_path.name}: {emb_path}")

    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    emb = np.load(emb_path)

    if emb.shape[0] != len(meta):
        raise ValueError(
            f"Row mismatch for {meta_path.name}: metadata has {len(meta)} rows, embeddings have {emb.shape[0]}"
        )
    return meta, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", type=str, default=str(DEFAULT_SHARD_DIR))
    ap.add_argument("--meta_out", type=str, default=str(DEFAULT_META_OUT))
    ap.add_argument("--emb_out", type=str, default=str(DEFAULT_EMB_OUT))
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    meta_paths = sorted(shard_dir.glob("*_metadata.parquet"))
    if not meta_paths:
        raise FileNotFoundError(f"No shard metadata files found in {shard_dir}")

    metas = []
    embs = []
    for meta_path in meta_paths:
        meta, emb = load_shard(meta_path)
        shard_name = meta_path.name.replace("_metadata.parquet", "")
        meta = meta.copy()
        meta["shard"] = shard_name
        metas.append(meta)
        embs.append(emb)
        print(f"[merge] {shard_name}: rows={len(meta):,} emb_shape={emb.shape}")

    merged_meta = pd.concat(metas, ignore_index=True)
    merged_emb = np.concatenate(embs, axis=0)

    if merged_emb.shape[0] != len(merged_meta):
        raise ValueError("Merged metadata/embedding row mismatch after concatenation")

    dup_cols = [c for c in ["building_id", "year", "city"] if c in merged_meta.columns]
    if dup_cols:
        dup_mask = merged_meta.duplicated(subset=dup_cols, keep=False)
        if dup_mask.any():
            sample = merged_meta.loc[dup_mask, dup_cols].head(10).to_dict("records")
            raise ValueError(
                f"Duplicate building-year keys detected across shards on columns {dup_cols}. "
                f"Sample duplicates: {sample}"
            )

    merged_meta = merged_meta.reset_index(drop=True)
    merged_meta["row_id"] = np.arange(len(merged_meta), dtype=np.int64)

    meta_out = Path(args.meta_out)
    emb_out = Path(args.emb_out)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    tmp_meta = meta_out.with_suffix(".parquet.tmp")
    tmp_emb = emb_out.with_suffix(".npy.tmp")
    merged_meta.to_parquet(tmp_meta, index=False)
    with open(tmp_emb, "wb") as f:
        np.save(f, merged_emb)
    tmp_meta.replace(meta_out)
    tmp_emb.replace(emb_out)

    print(f"[merge] wrote metadata → {meta_out}")
    print(f"[merge] wrote embeddings → {emb_out}  shape={merged_emb.shape}")
    if "status" in merged_meta.columns:
        print("[merge] status breakdown:")
        print(merged_meta["status"].value_counts().to_string())
    else:
        print("[merge] no status column present in merged metadata")


if __name__ == "__main__":
    main()
