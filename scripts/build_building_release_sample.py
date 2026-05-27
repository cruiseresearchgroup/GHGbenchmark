#!/usr/bin/env python3
"""Build a reviewer-friendly building benchmark sample package.

This script creates a small, representative subset of the public building
benchmark release so NeurIPS reviewers can inspect schema, data quality, and
the S2 multimodal linkage without downloading the full 1.6GB embedding file.

Outputs are written under ``paper_prep/dataset_release/building_sample/``:

- building_main_sample.csv
- building_year_climate_sample.csv
- building_year_s2_metadata_sample.parquet
- building_year_s2_embeddings_sample.npy
- sample_manifest.json
- README.md

The sample is intentionally:
- cross-region (US + AU + SG)
- multi-city
- multi-year
- small enough for reviewer inspection
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "paper_prep" / "dataset_release" / "building_sample"

BUILDING_CSV = PROCESSED_DIR / "building_all_aus_merged.csv"
CLIMATE_CSV = PROCESSED_DIR / "building_year_climate.csv"
S2_META = PROCESSED_DIR / "building_year_s2_metadata.parquet"
S2_EMB = PROCESSED_DIR / "building_year_s2_embeddings.npy"

SAMPLE_CITIES = [
    "nyc",
    "la",
    "chicago",
    "dc",
    "sf",
    "sydney",
    "melbourne",
    "singapore",
]
PER_CITY_CAP = 1200
RNG_SEED = 20260502


def _sample_city(df_city: pd.DataFrame, per_city_cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample approximately evenly across available years within a city."""
    n = len(df_city)
    if n <= per_city_cap:
        return df_city.copy()

    pieces = []
    years = sorted(df_city["year"].dropna().unique())
    quota = max(1, per_city_cap // max(1, len(years)))

    for year in years:
        df_y = df_city[df_city["year"] == year]
        take = min(len(df_y), quota)
        if take > 0:
            idx = rng.choice(df_y.index.to_numpy(), size=take, replace=False)
            pieces.append(df_city.loc[idx])

    out = pd.concat(pieces, ignore_index=False).drop_duplicates()
    if len(out) < per_city_cap:
        remain = df_city.drop(index=out.index, errors="ignore")
        need = min(per_city_cap - len(out), len(remain))
        if need > 0:
            idx = rng.choice(remain.index.to_numpy(), size=need, replace=False)
            out = pd.concat([out, df_city.loc[idx]], ignore_index=False)

    if len(out) > per_city_cap:
        idx = rng.choice(out.index.to_numpy(), size=per_city_cap, replace=False)
        out = out.loc[idx]

    return out.sort_values(["year", "building_id"]).reset_index(drop=True)


def _write_sample_readme(manifest: dict) -> None:
    text = f"""# GHGbench Building Sample

This directory contains a reviewer-friendly sample of the public **building**
portion of GHGbench.

It is a representative subset of the full building benchmark release and is
intended for:

- schema inspection
- basic data quality checks
- verifying the tabular + climate + S2 multimodal linkage

## Sampling protocol

- Cities included: {", ".join(manifest["sample_cities"])}
- Sampling seed: {manifest["sampling_seed"]}
- Per-city cap: {manifest["per_city_cap"]}
- Rows sampled from the canonical building table: {manifest["building_rows"]}
- Rows with climate records: {manifest["climate_rows"]}
- Rows with successful building-patch S2 metadata: {manifest["s2_metadata_rows"]}
- Embedding matrix shape: {tuple(manifest["s2_embeddings_shape"])}

The sample is stratified by city and approximately balanced across the years
available within each selected city.

## Files

- `building_main_sample.csv`
  - Subset of the canonical building table
- `building_year_climate_sample.csv`
  - Climate rows matched on `(city, building_id, year)`
- `building_year_s2_metadata_sample.parquet`
  - S2 metadata rows with `status == ok` and `img_source == building_patch`
- `building_year_s2_embeddings_sample.npy`
  - Embedding rows aligned to the metadata file row-for-row
- `sample_manifest.json`
  - Sampling statistics and provenance

## Notes

- This sample is for reviewer inspection and does **not** replace the full
  building release.
- The company-level benchmark is not redistributed here due to upstream data
  licensing constraints; see the main dataset landing page and code repository
  for reconstruction instructions.
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    building = pd.read_csv(BUILDING_CSV)
    building = building[building["city"].isin(SAMPLE_CITIES)].copy()

    sampled = []
    for city in SAMPLE_CITIES:
        df_city = building[building["city"] == city].copy()
        if df_city.empty:
            continue
        sampled.append(_sample_city(df_city, PER_CITY_CAP, rng))

    sample_df = pd.concat(sampled, ignore_index=True)
    sample_df = sample_df.drop_duplicates(subset=["city", "building_id", "year"]).reset_index(drop=True)

    key_cols = ["city", "building_id", "year"]
    sample_keys = sample_df[key_cols].copy()

    climate = pd.read_csv(CLIMATE_CSV)
    climate_sample = climate.merge(sample_keys, on=key_cols, how="inner")

    meta = pd.read_parquet(S2_META)
    meta = meta[
        meta["city"].isin(SAMPLE_CITIES)
        & meta["status"].eq("ok")
        & meta["img_source"].eq("building_patch")
    ].copy()
    meta = meta.reset_index(drop=False).rename(columns={"index": "embedding_row"})
    meta_sample = meta.merge(sample_keys, on=key_cols, how="inner")
    emb = np.load(S2_EMB, mmap_mode="r")
    emb_sample = np.asarray(emb[meta_sample["embedding_row"].to_numpy()])

    sample_df.to_csv(OUT_DIR / "building_main_sample.csv", index=False)
    climate_sample.to_csv(OUT_DIR / "building_year_climate_sample.csv", index=False)
    meta_sample.to_parquet(OUT_DIR / "building_year_s2_metadata_sample.parquet", index=False)
    np.save(OUT_DIR / "building_year_s2_embeddings_sample.npy", emb_sample)

    manifest = {
        "source_building_csv": str(BUILDING_CSV.relative_to(BASE_DIR)),
        "source_climate_csv": str(CLIMATE_CSV.relative_to(BASE_DIR)),
        "source_s2_metadata": str(S2_META.relative_to(BASE_DIR)),
        "source_s2_embeddings": str(S2_EMB.relative_to(BASE_DIR)),
        "sample_cities": SAMPLE_CITIES,
        "sampling_seed": RNG_SEED,
        "per_city_cap": PER_CITY_CAP,
        "building_rows": int(len(sample_df)),
        "climate_rows": int(len(climate_sample)),
        "s2_metadata_rows": int(len(meta_sample)),
        "s2_embeddings_shape": list(map(int, emb_sample.shape)),
        "city_counts": {k: int(v) for k, v in sample_df["city"].value_counts().sort_index().items()},
        "year_range": {
            "min": int(sample_df["year"].min()),
            "max": int(sample_df["year"].max()),
        },
    }
    (OUT_DIR / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_sample_readme(manifest)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
