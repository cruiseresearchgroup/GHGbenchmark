#!/usr/bin/env python
"""Backfill missing building coordinates from the same building in other years."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill missing lat/lon from the same building_id across years."
    )
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--city", default=None, help="Optional city filter, e.g. nyc.")
    p.add_argument(
        "--building-id-col",
        default="building_id",
        help="Building id column name.",
    )
    p.add_argument("--lat-col", default="latitude", help="Latitude column name.")
    p.add_argument("--lon-col", default="longitude", help="Longitude column name.")
    p.add_argument(
        "--coord-source-col",
        default="coord_source",
        help="Optional coordinate source column name.",
    )
    p.add_argument(
        "--filled-source-value",
        default="history_backfill",
        help="Source label to assign when coordinates are filled.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)
    work = df if args.city is None else df[df["city"] == args.city].copy()

    lat_col = args.lat_col
    lon_col = args.lon_col
    bid_col = args.building_id_col
    src_col = args.coord_source_col

    before_missing = (~(work[lat_col].notna() & work[lon_col].notna())).sum()

    has_coord = work[lat_col].notna() & work[lon_col].notna()
    coord_lookup = (
        work.loc[has_coord, [bid_col, lat_col, lon_col]]
        .drop_duplicates(subset=[bid_col])
        .set_index(bid_col)
    )

    miss_mask = ~(work[lat_col].notna() & work[lon_col].notna())
    fill_lat = work.loc[miss_mask, bid_col].map(coord_lookup[lat_col])
    fill_lon = work.loc[miss_mask, bid_col].map(coord_lookup[lon_col])
    can_fill = fill_lat.notna() & fill_lon.notna()
    fill_index = work.loc[miss_mask].index[can_fill.to_numpy()]

    work.loc[miss_mask, lat_col] = work.loc[miss_mask, lat_col].fillna(fill_lat)
    work.loc[miss_mask, lon_col] = work.loc[miss_mask, lon_col].fillna(fill_lon)
    if src_col in work.columns and len(fill_index) > 0:
        work.loc[fill_index, src_col] = args.filled_source_value

    after_missing = (~(work[lat_col].notna() & work[lon_col].notna())).sum()

    if args.city is None:
        out = work
    else:
        out = df.copy()
        cols_to_copy = [lat_col, lon_col]
        if src_col in work.columns and src_col in out.columns:
            cols_to_copy.append(src_col)
        out.loc[work.index, cols_to_copy] = work[cols_to_copy]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"city={args.city or 'ALL'}")
    print(f"before_missing={int(before_missing)}")
    print(f"fillable={int(can_fill.sum())}")
    print(f"after_missing={int(after_missing)}")


if __name__ == "__main__":
    main()
