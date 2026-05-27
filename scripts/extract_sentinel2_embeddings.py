"""End-to-end Sentinel-2 → Clay embedding pipeline for T2-MM.

One-shot: GEE annual cloud-free composite → 128×128 UTM patch → Clay v1.5 MAE
encoder → parquet row. Patches are held in RAM only; nothing is written to
disk except the final parquet of embeddings.

Per building-year:
  1. If building lat/lon passes ±2° sanity filter → fetch patch at that point
     (img_source='building_patch'). Otherwise fetch patch at the city centre
     (img_source='city_fallback'). Matches Phase-2 ERA5 extraction semantics.
  2. Patches with the same (round(lat,4), round(lon,4), year) triple share
     one GEE fetch + encoder call, then fan out to all sharing building-years.
  3. Annual median of cloud-masked L2A scenes (SCL-based mask). Fetched via
     ee.data.computePixels at exactly 128×128 in the local UTM zone,
     guaranteeing 10 m GSD for every band (20 m bands are resampled server
     side). Returns raw reflectance (divided by 10000).
  4. Clay v1.5 encoder runs on GPU in batches. Embedding = mean of unmasked
     patch tokens (one vector per image). Dimension depends on the exact
     Clay checkpoint (v1.5 ViT-L is 1024).

Year window: 2015–2025 only. Sentinel-2 L2A is not available before 2017
globally (2015 for Europe), so 2015/2016 rows may produce status='no_images'
for cities outside Europe — we still attempt and record the status honestly.

Output (two aligned files):
  data/processed/building_year_s2_metadata.parquet
    row_id, building_id, year, city, img_source, lat, lon,
    status ('ok' | 'no_images' | 'ee_error:...' | 'encode_error:...'),
    n_images (count of scenes in the annual composite)
  data/processed/building_year_s2_embeddings.npy
    (N, D) float32 matrix — row i corresponds to metadata row_id == i.
    Rows with status != 'ok' have all-zero vectors.
Split rationale: embedding is dense and fixed-shape, ~3 KB/row × ~1 M rows →
~3 GB. Keeping it out of the parquet avoids per-row Python list serialization
and gives downstream code a memmap-friendly tensor.

Prereqs:
  pip install earthengine-api pyproj pyarrow torch torchvision
  pip install git+https://github.com/Clay-foundation/model.git   # clay v1.5
  earthengine authenticate          # one-time, opens browser
  huggingface-cli login             # if Clay weights gated (v1.5 isn't)

Usage:
  python scripts/extract_sentinel2_embeddings.py                       # all
  python scripts/extract_sentinel2_embeddings.py --cities nyc,la
  python scripts/extract_sentinel2_embeddings.py --dry_run             # plan only

IMPORTANT — things this script assumes and that you should verify:
  * ee.data.computePixels availability. Modern (>=0.1.370) earthengine-api
    high-volume API. If it raises 'method not found', upgrade the package.
  * Clay v1.5 MAE API. The exact class/module path can drift release to
    release. Look at the ClayEncoder class below — if import fails, read
    https://github.com/Clay-foundation/model and adjust.
  * GEE daily quota. Research accounts usually fine; free tier will hit
    limits on a multi-hundred-thousand-patch run. The script is resume-safe
    so you can rerun after a quota reset.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    print("ERROR: earthengine-api not installed. Run: pip install earthengine-api", file=sys.stderr)
    sys.exit(1)

try:
    from pyproj import Transformer
except ImportError:
    print("ERROR: pyproj not installed. Run: pip install pyproj", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch torchvision", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_openmeteo_climate import CITY_CENTRES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
BUILDING_CSV = PROCESSED_DIR / 'building_all_aus_merged.csv'
META_PATH = PROCESSED_DIR / 'building_year_s2_metadata.parquet'
EMB_PATH = PROCESSED_DIR / 'building_year_s2_embeddings.npy'
MODEL_DIR = BASE_DIR / 'data' / 'models'

YEAR_MIN, YEAR_MAX = 2017, 2025  # S2 L2A global coverage starts 2017
SANITY_RADIUS_DEG = 2.0

PATCH_PX = 128
GSD = 10  # metres
PATCH_M = PATCH_PX * GSD  # 1280 m span

S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
# Central wavelengths (nm) for Sentinel-2A — Clay uses these for positional
# encoding of spectral bands. (S2B values differ by <2 nm, negligible.)
S2_WAVELENGTHS_NM = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8,
                     832.8, 864.7, 1613.7, 2202.4]
S2_SR_SCALE = 10000.0

DEDUP_LATLON_DECIMALS = 4  # ~11 m at the equator — practical cache key

COORD_FETCH_WORKERS = 8
CLAY_BATCH_SIZE = 32

# SCL classes we KEEP: 2=dark, 4=vegetation, 5=not-vegetated, 6=water,
# 7=unclassified, 11=snow. We drop 3=shadow, 8/9=cloud med/high, 10=cirrus.
SCL_KEEP = [2, 4, 5, 6, 7, 11]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_valid_coord(lat, lon, city_lat, city_lon):
    if pd.isna(lat) or pd.isna(lon):
        return False
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return False
    if lat == 0.0 or lon == 0.0:
        return False
    if abs(lat - city_lat) > SANITY_RADIUS_DEG:
        return False
    if abs(lon - city_lon) > SANITY_RADIUS_DEG:
        return False
    return True


def utm_epsg(lat, lon):
    zone = int((lon + 180.0) / 6.0) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


_TRANSFORMER_CACHE = {}

def lonlat_to_utm(lon, lat, epsg):
    if epsg not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[epsg] = Transformer.from_crs(
            'EPSG:4326', f'EPSG:{epsg}', always_xy=True
        )
    return _TRANSFORMER_CACHE[epsg].transform(lon, lat)


def scl_mask(img):
    scl = img.select('SCL')
    mask = scl.eq(SCL_KEEP[0])
    for v in SCL_KEEP[1:]:
        mask = mask.Or(scl.eq(v))
    return img.updateMask(mask)


# ---------------------------------------------------------------------------
# GEE fetch
# ---------------------------------------------------------------------------
def build_annual_collection(year, region):
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .filterBounds(region)
        .map(scl_mask)
    )


def fetch_patch(lat, lon, year):
    """Return (pixels (C, H, W) float32, n_images, status).

    pixels is None on any failure. n_images is 0 on no_images.
    """
    epsg = utm_epsg(lat, lon)
    x, y = lonlat_to_utm(lon, lat, epsg)

    region = ee.Geometry.Rectangle(
        [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]  # rough EPSG:4326 box for filterBounds
    )
    col = build_annual_collection(year, region).select(S2_BANDS)
    try:
        n_images = int(col.size().getInfo())
    except Exception as e:
        return None, 0, f'ee_error:size:{type(e).__name__}'
    if n_images == 0:
        return None, 0, 'no_images'

    img = col.median()

    request = {
        'expression': img,
        'fileFormat': 'NUMPY_NDARRAY',
        'bandIds': S2_BANDS,
        'grid': {
            'dimensions': {'width': PATCH_PX, 'height': PATCH_PX},
            'affineTransform': {
                'scaleX': GSD,
                'shearX': 0.0,
                'translateX': x - PATCH_M / 2.0,
                'shearY': 0.0,
                'scaleY': -GSD,
                'translateY': y + PATCH_M / 2.0,
            },
            'crsCode': f'EPSG:{epsg}',
        },
    }
    try:
        result = ee.data.computePixels(request)
    except Exception as e:
        return None, n_images, f'ee_error:pixels:{type(e).__name__}:{str(e)[:60]}'

    # result is a (H, W) structured ndarray with one field per band.
    try:
        arr = np.stack([result[b].astype(np.float32) for b in S2_BANDS], axis=0)
    except Exception as e:
        return None, n_images, f'ee_error:stack:{type(e).__name__}'
    arr = arr / S2_SR_SCALE  # raw SR → reflectance 0..1
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, n_images, 'ok'


# ---------------------------------------------------------------------------
# Clay encoder wrapper
# ---------------------------------------------------------------------------
class ClayEncoder:
    """Thin wrapper around Clay v1.5 MAE encoder.

    Clay's API has shifted across releases; if the import below fails, check
    https://github.com/Clay-foundation/model for the current entrypoint and
    adjust this class. The behaviour we rely on:
      * load a checkpoint from disk / hub
      * forward a dict with pixels, time, latlon, gsd, waves
      * return token embeddings we can mean-pool to a single vector per image
    """

    def __init__(self, ckpt_path, metadata_path, device):
        self.device = device
        # Clay v1.5 public entrypoint. If this import fails, the most common
        # alternative is `from src.module import ClayMAEModule` after cloning
        # the repo and adding its root to PYTHONPATH.
        try:
            from claymodel.module import ClayMAEModule
        except ImportError as e:
            raise ImportError(
                "Could not import claymodel. Install via:\n"
                "  pip install git+https://github.com/Clay-foundation/model.git\n"
                "or clone the repo and add its src to PYTHONPATH.\n"
                f"Original error: {e}"
            )
        module = ClayMAEModule.load_from_checkpoint(
            ckpt_path, map_location=device, strict=False, metadata_path=metadata_path
        )
        module.eval()
        self.model = module.to(device)
        self.waves = torch.tensor(S2_WAVELENGTHS_NM, device=device, dtype=torch.float32)
        self.gsd = torch.tensor(GSD, device=device, dtype=torch.float32)

    @staticmethod
    def _time_feat(year):
        # Clay expects (week, hour) sin/cos. We have annual medians, so use
        # mid-year (week 26, hour 12) as a neutral placeholder for every row.
        week = 26.0
        hour = 12.0
        w = 2 * np.pi * week / 52.0
        h = 2 * np.pi * hour / 24.0
        return [np.sin(w), np.cos(w), np.sin(h), np.cos(h)]

    @staticmethod
    def _latlon_feat(lat, lon):
        la = np.radians(lat)
        lo = np.radians(lon)
        return [np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)]

    @torch.no_grad()
    def encode_batch(self, pixels_np, lats, lons, years):
        """pixels_np: (B, C, H, W) float32. Returns (B, D) float32 numpy."""
        pixels = torch.from_numpy(pixels_np).to(self.device)
        times = torch.tensor(
            [self._time_feat(y) for y in years],
            device=self.device, dtype=torch.float32,
        )
        latlons = torch.tensor(
            [self._latlon_feat(la, lo) for la, lo in zip(lats, lons)],
            device=self.device, dtype=torch.float32,
        )
        datacube = {
            'platform': 'sentinel-2-l2a',
            'time': times,
            'latlon': latlons,
            'pixels': pixels,
            'gsd': self.gsd,
            'waves': self.waves,
        }
        out = self.model.model.encoder(datacube)
        # Clay MAE returns a tuple whose first element is (B, N+1, D) patch
        # tokens (index 0 is CLS). Mean-pool patch tokens for a robust vector.
        if isinstance(out, tuple):
            tokens = out[0]
        else:
            tokens = out
        if tokens.ndim == 3:
            emb = tokens[:, 1:, :].mean(dim=1)
        else:
            emb = tokens
        return emb.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_tasks(cities_filter, building_csv):
    df = pd.read_csv(
        building_csv,
        low_memory=False,
        usecols=['building_id', 'city', 'year', 'latitude', 'longitude'],
    )
    df = df[df['year'].between(YEAR_MIN, YEAR_MAX)].copy()
    df['year'] = df['year'].astype(int)
    if cities_filter:
        df = df[df['city'].isin(cities_filter)]

    tasks = []
    for row in df.itertuples(index=False):
        city = row.city
        if city not in CITY_CENTRES:
            continue
        c_lat, c_lon = CITY_CENTRES[city]
        if is_valid_coord(row.latitude, row.longitude, c_lat, c_lon):
            src = 'building_patch'
            lat, lon = float(row.latitude), float(row.longitude)
        else:
            src = 'city_fallback'
            lat, lon = c_lat, c_lon
        tasks.append({
            'building_id': row.building_id,
            'year': int(row.year),
            'city': city,
            'img_source': src,
            'lat': lat,
            'lon': lon,
        })
    return pd.DataFrame(tasks)


def load_existing(meta_path, emb_path):
    """Return (metadata DataFrame without row_id, embedding np.ndarray, done_set).

    done_set contains (building_id, year) tuples with status == 'ok'.
    Rows are returned in row_id order; the caller will re-index by appending.
    """
    if not meta_path.exists():
        return pd.DataFrame(), np.zeros((0, 0), dtype=np.float32), set()
    meta = pd.read_parquet(meta_path).sort_values('row_id').reset_index(drop=True)
    if emb_path.exists():
        emb = np.load(emb_path)
        if emb.shape[0] != len(meta):
            print(
                f'WARN: embedding file rows ({emb.shape[0]}) != metadata rows '
                f'({len(meta)}); rebuilding zero-dim matrix',
                file=sys.stderr,
            )
            emb = np.zeros((len(meta), 0), dtype=np.float32)
    else:
        emb = np.zeros((len(meta), 0), dtype=np.float32)
    done = set(
        (r.building_id, int(r.year))
        for r in meta.itertuples(index=False)
        if getattr(r, 'status', '') == 'ok'
    )
    return meta, emb, done


def flush(meta_rows, emb_matrix, meta_path, emb_path):
    """Atomic two-file flush.

    meta_rows: list[dict] (will be converted to DataFrame; row_id already set)
    emb_matrix: np.ndarray (N, D) aligned to meta_rows by row_id
    """
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame(meta_rows)
    tmp_meta = meta_path.with_suffix('.parquet.tmp')
    tmp_emb = emb_path.with_suffix('.npy.tmp')
    meta_df.to_parquet(tmp_meta, index=False)
    # np.save auto-appends ".npy" for string paths; write via file handle so the
    # temporary filename stays exactly what os.replace expects.
    with open(tmp_emb, 'wb') as f:
        np.save(f, emb_matrix)
    os.replace(tmp_meta, meta_path)
    os.replace(tmp_emb, emb_path)


def main():
    global YEAR_MIN, YEAR_MAX

    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', type=str, default=None)
    ap.add_argument(
        '--building_csv',
        type=str,
        default=str(BUILDING_CSV),
        help='Path to processed building CSV (default: canonical building_all_aus_merged.csv)',
    )
    ap.add_argument('--meta_out', type=str, default=str(META_PATH))
    ap.add_argument('--emb_out', type=str, default=str(EMB_PATH))
    ap.add_argument('--year_min', type=int, default=YEAR_MIN,
                    help=f'Min year (default {YEAR_MIN}; global S2 L2A starts 2017)')
    ap.add_argument('--year_max', type=int, default=YEAR_MAX)
    ap.add_argument('--batch_size', type=int, default=CLAY_BATCH_SIZE)
    ap.add_argument('--workers', type=int, default=COORD_FETCH_WORKERS)
    ap.add_argument('--device', type=str, default=None,
                    help='cuda / cpu (default: auto)')
    ap.add_argument('--clay_ckpt', type=str,
                    default=os.environ.get('CLAY_CKPT', 'clay-v1.5.ckpt'),
                    help='Path to Clay v1.5 checkpoint file')
    ap.add_argument('--clay_metadata', type=str,
                    default=os.environ.get('CLAY_METADATA', str(MODEL_DIR / 'clay_metadata.yaml')),
                    help='Path to Clay metadata.yaml (required because the pip package omits configs/metadata.yaml)')
    ap.add_argument('--ee_project', type=str,
                    default=os.environ.get('EE_PROJECT', 'earth-engine-493114'),
                    help='Google Earth Engine Cloud project ID')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    cities_filter = args.cities.split(',') if args.cities else None
    meta_path = Path(args.meta_out)
    emb_path = Path(args.emb_out)

    # Allow per-invocation override of the global year window.
    YEAR_MIN, YEAR_MAX = args.year_min, args.year_max

    building_csv = Path(args.building_csv)
    print(f'Loading buildings from {building_csv} ...')
    tasks_df = load_tasks(cities_filter, building_csv)
    print(f'  {len(tasks_df):,} building-year rows in {YEAR_MIN}–{YEAR_MAX}')
    print(f'  img_source: {tasks_df["img_source"].value_counts().to_dict()}')

    existing_meta, existing_emb, done = load_existing(meta_path, emb_path)
    print(f'  existing metadata rows: {len(existing_meta):,}  (ok={len(done):,})')

    todo = tasks_df[~tasks_df.apply(
        lambda r: (r['building_id'], r['year']) in done, axis=1
    )].copy()
    print(f'  pending: {len(todo):,}')
    if todo.empty:
        print('Nothing to do.')
        return

    # Dedup: shared GEE+encoder call per (rounded_lat, rounded_lon, year).
    todo['key_lat'] = todo['lat'].round(DEDUP_LATLON_DECIMALS)
    todo['key_lon'] = todo['lon'].round(DEDUP_LATLON_DECIMALS)
    unique_cache_keys = todo.groupby(['key_lat', 'key_lon', 'year']).size()
    print(f'  unique (lat,lon,year) cache keys: {len(unique_cache_keys):,}')
    print(f'  avg dedup factor: {len(todo) / max(len(unique_cache_keys), 1):.2f}x')

    if args.dry_run:
        print('[dry_run] exiting before any GEE / encoder work.')
        return

    # Init GEE — modern API requires a Cloud project ID.
    print(f'Initializing Earth Engine (project={args.ee_project}) ...')
    try:
        ee.Initialize(project=args.ee_project)
    except Exception:
        print('  ee.Initialize() failed. Running `earthengine authenticate` ...')
        ee.Authenticate()
        ee.Initialize(project=args.ee_project)

    # Init Clay
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading Clay encoder on {device} from {args.clay_ckpt} ...')
    print(f'Using Clay metadata: {args.clay_metadata}')
    encoder = ClayEncoder(args.clay_ckpt, args.clay_metadata, device)

    # Embedding cache: dedup key → np.ndarray or failure status string
    emb_cache = {}
    status_cache = {}
    n_images_cache = {}

    # Group tasks by cache key so one GEE call fans out to all matching rows.
    groups = todo.groupby(['key_lat', 'key_lon', 'year'])
    group_list = list(groups)
    print(f'Processing {len(group_list):,} unique fetches across {len(todo):,} rows ...')

    # Seed meta_rows / emb_list from existing data (keep row_id stable).
    meta_rows = existing_meta.to_dict('records') if not existing_meta.empty else []
    emb_list = [existing_emb[i] for i in range(existing_emb.shape[0])] if existing_emb.size else []
    emb_dim = existing_emb.shape[1] if existing_emb.size else None
    t0 = time.time()
    processed = 0

    # Process in chunks: fetch N patches in parallel → batch encode → attach.
    CHUNK = max(args.batch_size * 2, 32)
    for chunk_start in range(0, len(group_list), CHUNK):
        chunk = group_list[chunk_start: chunk_start + CHUNK]

        # Parallel GEE fetch
        fetched = {}
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            fut_to_key = {}
            for key, sub in chunk:
                klat, klon, kyear = key
                # pick any row from the group — they share lat/lon/year by construction
                row0 = sub.iloc[0]
                fut = exe.submit(fetch_patch, float(row0['lat']), float(row0['lon']), int(kyear))
                fut_to_key[fut] = key
            for fut in as_completed(fut_to_key):
                key = fut_to_key[fut]
                try:
                    pixels, n_img, status = fut.result()
                except Exception as e:
                    pixels, n_img, status = None, 0, f'ee_error:future:{type(e).__name__}'
                fetched[key] = (pixels, n_img, status)

        # Batch encode successful fetches
        ok_keys = [k for k, (px, _, st) in fetched.items() if st == 'ok' and px is not None]
        for i in range(0, len(ok_keys), args.batch_size):
            batch_keys = ok_keys[i: i + args.batch_size]
            batch_pixels = np.stack([fetched[k][0] for k in batch_keys], axis=0)
            lats = [float(k[0]) for k in batch_keys]
            lons = [float(k[1]) for k in batch_keys]
            years = [int(k[2]) for k in batch_keys]
            try:
                embs = encoder.encode_batch(batch_pixels, lats, lons, years)
            except Exception as e:
                for k in batch_keys:
                    status_cache[k] = f'encode_error:{type(e).__name__}:{str(e)[:60]}'
                continue
            for k, e in zip(batch_keys, embs):
                emb_cache[k] = e.tolist()
                status_cache[k] = 'ok'
                n_images_cache[k] = fetched[k][1]

        # Record non-ok fetches
        for k, (px, n_img, st) in fetched.items():
            if st != 'ok':
                status_cache[k] = st
                n_images_cache[k] = n_img

        # Infer embedding dim from the first successful batch.
        if emb_dim is None:
            for k in ok_keys:
                if k in emb_cache:
                    emb_dim = len(emb_cache[k])
                    break

        # Fan out from cache to all building-year rows in these groups
        for key, sub in chunk:
            status = status_cache.get(key, 'missing')
            emb_vec = emb_cache.get(key)
            n_img = n_images_cache.get(key, 0)
            for row in sub.itertuples(index=False):
                row_id = len(meta_rows)
                meta_rows.append({
                    'row_id': row_id,
                    'building_id': row.building_id,
                    'year': int(row.year),
                    'city': row.city,
                    'img_source': row.img_source,
                    'lat': float(row.lat),
                    'lon': float(row.lon),
                    'status': status,
                    'n_images': int(n_img),
                })
                if emb_vec is not None and emb_dim is not None:
                    emb_list.append(np.asarray(emb_vec, dtype=np.float32))
                elif emb_dim is not None:
                    emb_list.append(np.zeros(emb_dim, dtype=np.float32))
                else:
                    emb_list.append(None)  # placeholder until first success
            processed += len(sub)

        # Backfill None placeholders once we know emb_dim.
        if emb_dim is not None:
            for i, v in enumerate(emb_list):
                if v is None:
                    emb_list[i] = np.zeros(emb_dim, dtype=np.float32)
            emb_matrix = np.stack(emb_list, axis=0)
        else:
            emb_matrix = np.zeros((len(meta_rows), 0), dtype=np.float32)

        flush(meta_rows, emb_matrix, meta_path, emb_path)
        dt = time.time() - t0
        rate = processed / max(dt, 1e-6)
        eta_sec = (len(todo) - processed) / max(rate, 1e-6)
        print(
            f'  [{chunk_start + len(chunk):>6}/{len(group_list)} groups | '
            f'{processed:,}/{len(todo):,} rows] '
            f'{rate:.1f} row/s  ETA ~{eta_sec/3600:.1f} h',
            flush=True,
        )

    meta_df = pd.DataFrame(meta_rows)
    print(f'\nWrote {len(meta_df):,} metadata rows → {meta_path}')
    if emb_dim is not None:
        print(f'Wrote {len(emb_list)}×{emb_dim} embedding matrix → {emb_path}')
    print('status breakdown:')
    print(meta_df['status'].value_counts().to_string())


if __name__ == '__main__':
    main()
