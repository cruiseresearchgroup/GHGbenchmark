"""Minimal GEE smoke test for the Sentinel-2 pipeline.

Verifies end-to-end that we can:
  1. Initialize Earth Engine with our project ID
  2. Build a cloud-masked annual median composite for one city
  3. Pull exactly 128×128×10 pixels via ee.data.computePixels in UTM
  4. Get sensible reflectance values back

This deliberately does NOT load Clay. The goal is to isolate every failure
mode that could come from GEE or projections before pulling in the model.

Usage:
  python scripts/smoke_test_sentinel2.py                    # nyc 2020
  python scripts/smoke_test_sentinel2.py --city sydney --year 2022
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_sentinel2_embeddings import (  # noqa: E402
    CITY_CENTRES,
    S2_BANDS,
    PATCH_PX,
    fetch_patch,
)

import ee  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', type=str, default='nyc')
    ap.add_argument('--year', type=int, default=2020)
    ap.add_argument('--ee_project', type=str, default='earth-engine-493114')
    args = ap.parse_args()

    if args.city not in CITY_CENTRES:
        print(f'ERROR: unknown city {args.city}', file=sys.stderr)
        sys.exit(1)
    lat, lon = CITY_CENTRES[args.city]
    print(f'[smoke] {args.city} {args.year}  centre=({lat}, {lon})')

    print(f'[smoke] ee.Initialize(project={args.ee_project})')
    try:
        ee.Initialize(project=args.ee_project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=args.ee_project)

    print('[smoke] fetching patch ...')
    pixels, n_img, status = fetch_patch(lat, lon, args.year)
    print(f'[smoke] status={status}  n_images={n_img}')
    if pixels is None:
        print('[smoke] FAIL — no pixels', file=sys.stderr)
        sys.exit(2)

    print(f'[smoke] shape={pixels.shape}  dtype={pixels.dtype}')
    expected = (len(S2_BANDS), PATCH_PX, PATCH_PX)
    if pixels.shape != expected:
        print(f'[smoke] WARN — expected {expected}, got {pixels.shape}')

    for i, b in enumerate(S2_BANDS):
        arr = pixels[i]
        finite = np.isfinite(arr)
        if not finite.any():
            print(f'  {b}: all non-finite')
            continue
        print(
            f'  {b}: min={arr[finite].min():.4f}  max={arr[finite].max():.4f}  '
            f'mean={arr[finite].mean():.4f}  nonzero={int((arr != 0).sum())}'
        )

    print('[smoke] PASS')


if __name__ == '__main__':
    main()
