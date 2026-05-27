"""
Standardize all building-level datasets into a unified schema.
Output: data/processed/building_all.csv

Unified columns:
    building_id, city, year, property_type, year_built, gross_floor_area_sqft,
    site_eui_kbtu_sqft, source_eui_kbtu_sqft, energy_star_score,
    total_ghg_emissions_mtco2e, electricity_kwh, natural_gas_kbtu,
    latitude, longitude
"""

import pandas as pd
import numpy as np
import json
import glob
import os
import warnings
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

UNIFIED_COLS = [
    'building_id', 'city', 'year', 'property_type', 'year_built', 'gross_floor_area_sqft',
    'site_eui_kbtu_sqft', 'source_eui_kbtu_sqft', 'energy_star_score',
    'total_ghg_emissions_mtco2e', 'electricity_kwh', 'natural_gas_kbtu',
    'latitude', 'longitude'
]

# Load property type mapping
with open(os.path.join(BASE_DIR, 'scripts', 'property_type_mapping.json')) as f:
    PT_MAP = json.load(f)['mapping']

# Build case-insensitive lookup
PT_MAP_LOWER = {k.strip().lower(): v for k, v in PT_MAP.items()}


def map_property_type(raw_type):
    """Map raw property type to unified category."""
    if pd.isna(raw_type):
        return np.nan
    raw = str(raw_type).strip()
    # Exact match first
    if raw in PT_MAP:
        return PT_MAP[raw]
    # Case-insensitive
    lower = raw.lower()
    if lower in PT_MAP_LOWER:
        return PT_MAP_LOWER[lower]
    # Partial matching for common patterns
    if 'office' in lower:
        return 'Office'
    if 'multifamily' in lower or 'apartment' in lower or 'apt ' in lower or 'residential' in lower:
        return 'Multifamily Housing'
    if 'hotel' in lower or 'motel' in lower:
        return 'Hotel'
    if 'school' in lower or 'k-12' in lower:
        return 'K-12 School'
    if 'college' in lower or 'university' in lower:
        return 'College/University'
    if 'hospital' in lower or 'medical' in lower or 'clinic' in lower:
        return 'Hospital/Medical'
    if 'warehouse' in lower or 'distribution' in lower:
        return 'Warehouse/Distribution'
    if 'retail' in lower or 'store' in lower or 'mall' in lower:
        return 'Retail'
    # Bare 'commercial' is too coarse to collapse into Retail (this was a
    # bug for SF, whose self-reported type 'Commercial' covers office +
    # retail + mixed). Route it to a dedicated mixed-use bucket.
    if lower in ('commercial', 'commercial - port facility') or lower.startswith('commercial '):
        return 'Commercial (Mixed)'
    if 'worship' in lower or 'church' in lower or 'synagogue' in lower or 'mosque' in lower:
        return 'Worship'
    if 'senior' in lower or 'nursing' in lower:
        return 'Senior Living'
    if 'parking' in lower or 'garage' in lower:
        return 'Parking'
    if 'restaurant' in lower or 'food service' in lower:
        return 'Restaurant'
    if 'supermarket' in lower or 'grocery' in lower:
        return 'Supermarket/Grocery'
    if 'manufacturing' in lower or 'industrial' in lower:
        return 'Industrial'
    if 'laboratory' in lower or 'lab' == lower:
        return 'Laboratory'
    if 'library' in lower:
        return 'Library'
    if 'fitness' in lower or 'gym' in lower:
        return 'Fitness Center'
    if 'mixed' in lower:
        return 'Mixed Use'
    if 'self-storage' in lower or 'storage' in lower:
        return 'Self-Storage'
    if 'dormitor' in lower or 'residence hall' in lower:
        return 'Residence Hall'
    if 'condo' in lower:
        return 'Multifamily Housing'
    return 'Other'


def to_numeric_safe(series):
    """Convert series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series.replace(['Not Available', 'N/A', 'NA', '', 'None', '-'], np.nan), errors='coerce')


def make_unified(df, city=None):
    """Ensure output has exactly the unified columns."""
    for col in UNIFIED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if city is not None:
        df['city'] = city
    return df[UNIFIED_COLS].copy()


def read_table_auto(path, **kwargs):
    """Read CSV/XLS/XLSX/XLSB with sensible defaults."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, low_memory=False, **kwargs)
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path, **kwargs)
    if ext == '.xlsb':
        try:
            return pd.read_excel(path, engine='pyxlsb', **kwargs)
        except ImportError as e:
            raise ImportError(
                f"Reading {os.path.basename(path)} requires pyxlsb. "
                "Install it with `pip install pyxlsb` or `conda install pyxlsb`."
            ) from e
    raise ValueError(f"Unsupported file extension: {ext}")


def read_excel_best_effort(path, sheet_names=None, header_candidates=(0, 1, 2)):
    """
    Try several sheets/header rows and pick the parse whose columns look most
    like a benchmarking table rather than data rows used as headers.
    """
    xls = pd.ExcelFile(path)
    if sheet_names:
        # Tolerate whitespace / case differences between the caller-provided
        # sheet name and the real sheet name in the workbook.
        wanted = {s.strip().lower(): s for s in sheet_names}
        matched = [real for real in xls.sheet_names
                   if real.strip().lower() in wanted]
        candidate_sheets = matched if matched else sheet_names
    else:
        candidate_sheets = xls.sheet_names

    keywords = (
        'property', 'gross', 'floor area', 'energy', 'ghg', 'year built',
        'parcel', 'berdo', 'site eui', 'source eui', 'score', 'address',
        'building', 'latitude', 'longitude'
    )
    best_score = -1
    best = None

    for sheet in candidate_sheets:
        for header in header_candidates:
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header)
            except Exception:
                continue
            cols = [str(c).strip() for c in df.columns]
            score = sum(any(k in c.lower() for k in keywords) for c in cols)
            if score > best_score:
                best_score = score
                best = df

    if best is None:
        raise ValueError(f"Could not parse any sheet from {path}")
    return best


def deduplicate_building_years(df):
    """
    Collapse duplicate city/building_id/year rows into a single building-year row.
    Numeric columns use the median of non-null values; categorical columns use mode.
    """
    key_cols = ['city', 'building_id', 'year']
    dup_rows = df.duplicated(subset=key_cols).sum()
    if dup_rows == 0:
        return df.reset_index(drop=True), 0

    agg = {}
    for col in df.columns:
        if col in key_cols:
            continue
        if is_numeric_dtype(df[col]):
            agg[col] = lambda s: np.nan if s.dropna().empty else float(np.nanmedian(s.dropna()))
        else:
            agg[col] = lambda s: np.nan if s.dropna().empty else s.dropna().mode().iloc[0]

    deduped = df.groupby(key_cols, dropna=False, as_index=False).agg(agg)
    return deduped.reset_index(drop=True), int(dup_rows)


def write_validation_reports(df):
    """Write lightweight validation artifacts for downstream inspection."""
    summary_rows = []
    availability_rows = []
    for city, city_df in df.groupby('city'):
        summary_rows.append({
            'city': city,
            'rows': len(city_df),
            'unique_buildings': city_df['building_id'].nunique(),
            'year_min': city_df['year'].min(),
            'year_max': city_df['year'].max(),
            'n_years': city_df['year'].nunique(dropna=True),
            'target_non_null': int(city_df['total_ghg_emissions_mtco2e'].notna().sum()),
            'site_eui_non_null': int(city_df['site_eui_kbtu_sqft'].notna().sum()),
            'duplicate_building_year_rows': int(city_df.duplicated(['building_id', 'year']).sum()),
            'fallback_row_ids': int(city_df['building_id'].astype(str).str.contains('_row_').sum()),
        })
        for col in UNIFIED_COLS:
            if col == 'city':
                continue
            availability_rows.append({
                'city': city,
                'column': col,
                'non_null_count': int(city_df[col].notna().sum()),
                'missing_rate_pct': float(city_df[col].isna().mean() * 100.0),
                'all_null': bool(city_df[col].notna().sum() == 0),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values('city')
    availability_df = pd.DataFrame(availability_rows).sort_values(['city', 'column'])

    summary_path = os.path.join(OUT_DIR, 'building_validation_summary.csv')
    availability_path = os.path.join(OUT_DIR, 'city_feature_availability.csv')
    summary_df.to_csv(summary_path, index=False)
    availability_df.to_csv(availability_path, index=False)

    print(f"\nValidation summary saved to {summary_path}")
    print(f"City-feature availability saved to {availability_path}")


def normalize_aus_city(suburb, state, address=None):
    """
    Normalize Australian suburb/state strings into a benchmark-friendly city label.
    Prefer major metro city names where the suburb is clearly part of a metro area;
    otherwise fall back to the cleaned suburb/city field.
    """
    suburb = '' if pd.isna(suburb) else str(suburb).strip().lower()
    state = '' if pd.isna(state) else str(state).strip().upper()
    address = '' if pd.isna(address) else str(address).strip().lower()

    suburb_clean = suburb.replace('/', ' ').replace('-', ' ')
    suburb_clean = ' '.join(suburb_clean.split())

    melbourne_metro = {
        'melbourne', 'docklands', 'southbank', 'south melbourne', 'west melbourne',
        'east melbourne', 'north melbourne', 'port melbourne', 'south wharf',
        'carlton', 'carlton south'
    }
    sydney_metro = {
        'sydney', 'north sydney', 'parramatta', 'st leonards', 'north ryde',
        'macquarie park', 'surry hills', 'barangaroo', 'pyrmont', 'chatswood'
    }
    brisbane_metro = {
        'brisbane', 'brisbane city', 'south brisbane', 'milton', 'newstead',
        'fortitude valley', 'spring hill', 'kangaroo point'
    }
    perth_metro = {'perth', 'west perth', 'east perth'}
    adelaide_metro = {'adelaide', 'north adelaide'}
    canberra_metro = {'canberra', 'city', 'civic', 'barton', 'braddon'}
    hobart_metro = {'hobart'}
    darwin_metro = {'darwin'}

    if suburb_clean in melbourne_metro or (' vic' in f' {address}' and suburb_clean in {'city'}):
        return 'melbourne'
    if suburb_clean in sydney_metro:
        return 'sydney'
    if suburb_clean in brisbane_metro:
        return 'brisbane'
    if suburb_clean in perth_metro:
        return 'perth'
    if suburb_clean in adelaide_metro:
        return 'adelaide'
    if suburb_clean in canberra_metro and state == 'ACT':
        return 'canberra'
    if suburb_clean in hobart_metro:
        return 'hobart'
    if suburb_clean in darwin_metro:
        return 'darwin'

    if suburb_clean:
        return suburb_clean.replace(' ', '_')

    state_fallback = {
        'VIC': 'melbourne',
        'NSW': 'sydney',
        'QLD': 'brisbane',
        'WA': 'perth',
        'SA': 'adelaide',
        'ACT': 'canberra',
        'TAS': 'hobart',
        'NT': 'darwin',
    }
    return state_fallback.get(state, 'australia_unknown')


# ============================================================
# Per-city processing functions
# ============================================================

def process_nyc():
    """NYC LL84: read CSV/XLSX/XLSB yearly files with a unified parser."""
    print("Processing NYC...")
    files = sorted(
        glob.glob(os.path.join(RAW_DIR, 'nyc_ll84', 'nyc_ll84_cy*.csv')) +
        glob.glob(os.path.join(RAW_DIR, 'nyc_ll84', 'nyc_ll84_cy*.xlsx')) +
        glob.glob(os.path.join(RAW_DIR, 'nyc_ll84', 'nyc_ll84_cy*.xlsb'))
    )
    dfs = []
    for f in files:
        try:
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.xlsx', '.xls'):
                df = read_excel_best_effort(
                    f,
                    sheet_names=['Information and Metrics', 'Data', 'Sheet1'],
                    header_candidates=(0, 1, 2),
                )
            elif ext == '.xlsb':
                # NYC 2016 is an .xlsb workbook whose first sheet is a README.
                # The actual records are split across several data sheets.
                sheet_dfs = []
                for sheet_name in ['Information and Metrics', 'Multi-BBL', 'Child no Parent', 'Not on CBL']:
                    try:
                        sheet_dfs.append(read_table_auto(f, sheet_name=sheet_name))
                    except ValueError:
                        continue
                if not sheet_dfs:
                    raise ValueError(f"No NYC data sheets found in {os.path.basename(f)}")
                df = pd.concat(sheet_dfs, ignore_index=True, sort=False)
            else:
                df = read_table_auto(f)
        except ImportError as e:
            print(f"  Warning: skipping {os.path.basename(f)}: {e}")
            continue
        # Column names vary slightly across years — find the right ones
        col_map = {}

        # Year — newer files have 'Calendar Year', older have 'Year Ending' (date)
        for c in df.columns:
            if 'calendar year' in c.lower():
                col_map['year'] = c
                break
        if 'year' not in col_map:
            # Extract year from filename as fallback
            for c in df.columns:
                if 'year ending' in c.lower():
                    col_map['_year_ending'] = c
                    break

        # Property type
        for c in df.columns:
            if 'primary property type' in c.lower() and 'self' in c.lower():
                col_map['property_type'] = c
                break
            elif 'largest property use type' in c.lower() and 'gross' not in c.lower():
                col_map['property_type'] = c

        # Year built
        for c in df.columns:
            if c.lower().strip() == 'year built':
                col_map['year_built'] = c
                break

        # Floor area - prefer self-reported GFA
        for c in df.columns:
            if 'property gfa' in c.lower() and 'self' in c.lower() and 'ft' in c.lower():
                col_map['gross_floor_area_sqft'] = c
                break
        if 'gross_floor_area_sqft' not in col_map:
            for c in df.columns:
                if 'dof gross floor area' in c.lower():
                    col_map['gross_floor_area_sqft'] = c
                    break

        # Site EUI
        for c in df.columns:
            if 'site eui' in c.lower() and 'weather' not in c.lower() and 'kbtu' in c.lower():
                col_map['site_eui_kbtu_sqft'] = c
                break

        # Source EUI
        for c in df.columns:
            if 'source eui' in c.lower() and 'weather' not in c.lower() and 'kbtu' in c.lower():
                col_map['source_eui_kbtu_sqft'] = c
                break

        # Energy Star
        for c in df.columns:
            if 'energy star score' in c.lower() and 'alert' not in c.lower():
                col_map['energy_star_score'] = c
                break

        # GHG - prefer total location-based
        for c in df.columns:
            if 'total' in c.lower() and 'ghg' in c.lower() and 'metric tons' in c.lower():
                col_map['total_ghg_emissions_mtco2e'] = c
                break

        # Electricity
        for c in df.columns:
            if 'electricity use' in c.lower() and 'grid' in c.lower() and 'kwh' in c.lower():
                col_map['electricity_kwh'] = c
                break

        # Natural gas
        for c in df.columns:
            if 'natural gas use' in c.lower() and 'kbtu' in c.lower():
                col_map['natural_gas_kbtu'] = c
                break

        # Lat/Lon
        for c in df.columns:
            if c.lower().strip() == 'latitude':
                col_map['latitude'] = c
            elif c.lower().strip() == 'longitude':
                col_map['longitude'] = c

        # Building ID — Property Id / Property ID
        for c in df.columns:
            if c.lower().strip() in ('property id', 'property id'):
                col_map['_raw_id'] = c
                break

        out = pd.DataFrame()
        out['city'] = ['nyc'] * len(df)

        # Generate building_id
        if '_raw_id' in col_map:
            out['building_id'] = 'nyc_' + df[col_map['_raw_id']].astype(str).str.strip()
        else:
            out['building_id'] = ['nyc_row_' + str(i) for i in range(len(df))]

        # Handle year
        if 'year' in col_map:
            out['year'] = df[col_map['year']].values
        elif '_year_ending' in col_map:
            # Extract year from date string like "12/31/2012"
            dates = pd.to_datetime(df[col_map['_year_ending']], errors='coerce')
            out['year'] = dates.dt.year.values
        else:
            # Extract from filename
            import re
            yr_match = re.search(r'cy(\d{4})', os.path.basename(f))
            out['year'] = int(yr_match.group(1)) if yr_match else np.nan

        for unified_name, raw_name in col_map.items():
            if unified_name.startswith('_'):
                continue  # skip internal keys
            if unified_name == 'year':
                continue  # already handled
            out[unified_name] = df[raw_name].values

        if 'property_type' in out.columns:
            out['property_type'] = out['property_type'].apply(map_property_type)
        else:
            out['property_type'] = np.nan

        for col in ['year', 'year_built', 'gross_floor_area_sqft', 'site_eui_kbtu_sqft',
                     'source_eui_kbtu_sqft', 'energy_star_score', 'total_ghg_emissions_mtco2e',
                     'electricity_kwh', 'natural_gas_kbtu', 'latitude', 'longitude']:
            if col in out.columns:
                out[col] = to_numeric_safe(out[col])

        dfs.append(make_unified(out, city='nyc'))

    result = pd.concat(dfs, ignore_index=True)

    # NYC 2016/2017 workbooks do not expose lat/lon columns in the same way as
    # later CSV years, but the same Property ID usually appears in other years
    # with valid coordinates. Reuse those building-level coordinates first
    # before falling back to any external geocoding/join workflow.
    has_coord = result['latitude'].notna() & result['longitude'].notna()
    if has_coord.any():
        coord_lookup = (
            result.loc[has_coord, ['building_id', 'latitude', 'longitude']]
            .drop_duplicates(subset=['building_id'])
            .set_index('building_id')
        )
        missing = result['latitude'].isna() | result['longitude'].isna()
        if missing.any():
            result.loc[missing, 'latitude'] = (
                result.loc[missing, 'building_id'].map(coord_lookup['latitude'])
            )
            result.loc[missing, 'longitude'] = (
                result.loc[missing, 'building_id'].map(coord_lookup['longitude'])
            )

    print(f"  NYC: {len(result)} rows, years {result['year'].min():.0f}-{result['year'].max():.0f}")
    return result


def process_chicago():
    """Chicago benchmarking."""
    print("Processing Chicago...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'chicago_benchmarking', 'chicago_benchmarking_all.csv'), low_memory=False)

    out = pd.DataFrame()
    out['building_id'] = 'chicago_' + df['ID'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['Data Year'])
    out['property_type'] = df['Primary Property Type'].apply(map_property_type)
    out['year_built'] = to_numeric_safe(df['Year Built'])
    out['gross_floor_area_sqft'] = to_numeric_safe(df['Gross Floor Area - Buildings (sq ft)'])
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df['Site EUI (kBtu/sq ft)'])
    out['source_eui_kbtu_sqft'] = to_numeric_safe(df['Source EUI (kBtu/sq ft)'])
    out['energy_star_score'] = to_numeric_safe(df['ENERGY STAR Score'])
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df['Total GHG Emissions (Metric Tons CO2e)'])
    # Chicago electricity and gas are in kBtu
    elec_kbtu = to_numeric_safe(df['Electricity Use (kBtu)'])
    out['electricity_kwh'] = elec_kbtu / 3.412  # kBtu -> kWh
    out['natural_gas_kbtu'] = to_numeric_safe(df['Natural Gas Use (kBtu)'])
    out['latitude'] = to_numeric_safe(df['Latitude'])
    out['longitude'] = to_numeric_safe(df['Longitude'])

    result = make_unified(out, city='chicago')
    print(f"  Chicago: {len(result)} rows")
    return result


def process_seattle():
    """Seattle benchmarking."""
    print("Processing Seattle...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'seattle_benchmarking', 'seattle_benchmarking_2015_present.csv'), low_memory=False)

    out = pd.DataFrame()
    out['building_id'] = 'seattle_' + df['OSEBuildingID'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['DataYear'])
    out['property_type'] = df['LargestPropertyUseType'].apply(map_property_type)
    out['year_built'] = to_numeric_safe(df['YearBuilt'])
    out['gross_floor_area_sqft'] = to_numeric_safe(df['PropertyGFATotal'])
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df['SiteEUI(kBtu/sf)'])
    out['source_eui_kbtu_sqft'] = to_numeric_safe(df['SourceEUI(kBtu/sf)'])
    out['energy_star_score'] = to_numeric_safe(df['ENERGYSTARScore'])
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df['TotalGHGEmissions'])
    out['electricity_kwh'] = to_numeric_safe(df['Electricity(kWh)'])
    # Seattle natural gas in therms -> kBtu (1 therm = 100 kBtu)
    out['natural_gas_kbtu'] = to_numeric_safe(df['NaturalGas(therms)']) * 100
    out['latitude'] = to_numeric_safe(df['Latitude'])
    out['longitude'] = to_numeric_safe(df['Longitude'])

    result = make_unified(out, city='seattle')
    print(f"  Seattle: {len(result)} rows")
    return result


def process_dc():
    """DC benchmarking."""
    print("Processing DC...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'dc_benchmarking', 'dc_benchmarking_all.csv'), low_memory=False)

    # Find the correct column names (may have slight variations)
    def find_col(patterns, columns):
        for p in patterns:
            for c in columns:
                if p.lower() in c.lower():
                    return c
        return None

    out = pd.DataFrame()
    out['building_id'] = 'dc_' + df['PID'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['REPORTINGYEAR'])

    pt_col = find_col(['PRIMARYPROPERTYTYPE_SELFSELECT'], df.columns)
    out['property_type'] = df[pt_col].apply(map_property_type) if pt_col else np.nan

    out['year_built'] = to_numeric_safe(df['YEARBUILT'])

    gfa_col = find_col(['REPORTEDBUILDINGGROSSFLOORAREA'], df.columns)
    out['gross_floor_area_sqft'] = to_numeric_safe(df[gfa_col]) if gfa_col else np.nan

    eui_col = find_col(['SITEEUI_KBTU_FT'], df.columns)
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df[eui_col]) if eui_col else np.nan

    seui_col = find_col(['SOURCEEUI_KBTU_FT'], df.columns)
    out['source_eui_kbtu_sqft'] = to_numeric_safe(df[seui_col]) if seui_col else np.nan

    out['energy_star_score'] = to_numeric_safe(df['ENERGYSTARSCORE'])

    ghg_col = find_col(['TOTGHGEMISSIONS_METRICTONSCO2E'], df.columns)
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df[ghg_col]) if ghg_col else np.nan

    # DC electricity in kWh
    elec_col = find_col(['ELECTRICITYUSE_GRID_KWH', 'ELECTRICITYUSE_KWH'], df.columns)
    out['electricity_kwh'] = to_numeric_safe(df[elec_col]) if elec_col else np.nan

    # DC natural gas in therms -> kBtu
    gas_col = find_col(['NATURALGASUSE_THERMS'], df.columns)
    out['natural_gas_kbtu'] = to_numeric_safe(df[gas_col]) * 100 if gas_col else np.nan

    out['latitude'] = to_numeric_safe(df['LATITUDE'])
    out['longitude'] = to_numeric_safe(df['LONGITUDE'])

    result = make_unified(out, city='dc')
    print(f"  DC: {len(result)} rows")
    return result


def process_sf():
    """SF benchmarking."""
    print("Processing SF...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'sf_benchmarking', 'sf_benchmarking_all.csv'), low_memory=False)

    out = pd.DataFrame()
    out['building_id'] = 'sf_' + df['Parcel Number'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['Benchmark Year'])
    out['property_type'] = df['Property Type - Self Selected'].apply(map_property_type)
    out['year_built'] = to_numeric_safe(df['Year Built'])
    out['gross_floor_area_sqft'] = to_numeric_safe(df['Floor Area'])

    # SF EUI column name has slightly different format
    eui_col = [c for c in df.columns if 'site eui' in c.lower() and 'weather' not in c.lower()][0]
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df[eui_col])

    seui_col = [c for c in df.columns if 'source eui' in c.lower() and 'weather' not in c.lower()][0]
    out['source_eui_kbtu_sqft'] = to_numeric_safe(df[seui_col])

    out['energy_star_score'] = to_numeric_safe(df['ENERGY STAR Score'])

    ghg_col = [c for c in df.columns if 'total ghg' in c.lower()][0]
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df[ghg_col])

    # SF electricity in kWh
    elec_col = [c for c in df.columns if 'electricity' in c.lower() and 'kwh' in c.lower()]
    out['electricity_kwh'] = to_numeric_safe(df[elec_col[0]]) if elec_col else np.nan

    # SF natural gas in kBtu
    gas_col = [c for c in df.columns if 'natural gas' in c.lower() and 'kbtu' in c.lower()]
    out['natural_gas_kbtu'] = to_numeric_safe(df[gas_col[0]]) if gas_col else np.nan

    out['latitude'] = to_numeric_safe(df['latitude'])
    out['longitude'] = to_numeric_safe(df['longitude'])

    result = make_unified(out, city='sf')
    print(f"  SF: {len(result)} rows")
    return result


def process_la():
    """LA EBEWE benchmarking."""
    print("Processing LA...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'la_benchmarking', 'la_ebewe_all.csv'), low_memory=False)

    out = pd.DataFrame()
    out['building_id'] = 'la_' + df['BUILDING ID'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['PROGRAM YEAR'])
    out['property_type'] = df['PROPERTY TYPE'].apply(map_property_type)
    out['year_built'] = to_numeric_safe(df['YEAR BUILT'])

    gfa_col = [c for c in df.columns if 'gross building floor area' in c.lower()][0]
    out['gross_floor_area_sqft'] = to_numeric_safe(df[gfa_col])

    # LA: be careful not to pick up "% DIFFERENCE FROM NATIONAL MEDIAN" columns
    eui_col = [c for c in df.columns if 'site' in c.lower() and 'eui' in c.lower()
               and 'weather' not in c.lower() and '%' not in c and 'difference' not in c.lower()][0]
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df[eui_col])

    seui_col = [c for c in df.columns if 'source' in c.lower() and 'eui' in c.lower()
                and 'weather' not in c.lower() and '%' not in c and 'difference' not in c.lower()]
    out['source_eui_kbtu_sqft'] = to_numeric_safe(df[seui_col[0]]) if seui_col else np.nan

    out['energy_star_score'] = to_numeric_safe(df['ENERGY STAR SCORE'])

    ghg_col = [c for c in df.columns if 'carbon dioxide' in c.lower() or 'co2' in c.lower()][0]
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df[ghg_col])

    # LA has no electricity/gas breakdown or lat/lon
    out['electricity_kwh'] = np.nan
    out['natural_gas_kbtu'] = np.nan
    out['latitude'] = np.nan
    out['longitude'] = np.nan

    result = make_unified(out, city='la')
    print(f"  LA: {len(result)} rows")
    return result


def process_boston():
    """Boston BERDO: 2 CSVs (2016-2017) + 7 XLSX (2018-2024)."""
    print("Processing Boston...")
    dfs = []

    # CSV files (2016-2017) — consistent format
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, 'boston_berdo', '*.csv')))
    for f in csv_files:
        year_str = os.path.basename(f).replace('berdo_cy', '').replace('.csv', '')
        df = pd.read_csv(f, encoding='latin1', low_memory=False)

        out = pd.DataFrame()
        out['building_id'] = 'boston_' + df['Tax Parcel'].astype(str).str.strip()
        out['property_type'] = df['Property Type'].apply(map_property_type)
        out['year'] = int(year_str)
        out['year_built'] = to_numeric_safe(df['Year Built'])
        out['gross_floor_area_sqft'] = to_numeric_safe(df[' Gross Area (sq ft) '])
        out['site_eui_kbtu_sqft'] = to_numeric_safe(df['Site EUI (kBTU/sf)'])
        out['source_eui_kbtu_sqft'] = np.nan
        out['energy_star_score'] = to_numeric_safe(df['Energy Star Score'])
        out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df[' GHG Emissions (MTCO2e) '])
        out['electricity_kwh'] = np.nan
        out['natural_gas_kbtu'] = np.nan
        out['latitude'] = np.nan
        out['longitude'] = np.nan

        dfs.append(make_unified(out, city='boston'))

    # XLSX files (2018-2024) — header row 1 (0-indexed)
    xlsx_files = sorted(glob.glob(os.path.join(RAW_DIR, 'boston_berdo', '*.xlsx')))
    for f in xlsx_files:
        year_str = os.path.basename(f).replace('berdo_cy', '').replace('.xlsx', '')
        try:
            df = read_excel_best_effort(
                f,
                sheet_names=['Data Disclosure', '2020 Data', 'Final Form', 'Sheet1',
                             'Data Disclosure - Campuses', '2024 Reported Energy and Water'],
                header_candidates=(0, 1, 2),
            )
        except Exception:
            try:
                df = read_excel_best_effort(f, header_candidates=(0, 1, 2))
            except Exception as e:
                print(f"  Warning: skipping {f}: {e}")
                continue

        out = pd.DataFrame()

        # Building ID — try BERDO ID, Tax Parcel ID, Tax Parcel
        bid_col = [c for c in df.columns if 'tax parcel' in str(c).lower() and 'unnamed' not in str(c).lower()]
        if not bid_col:
            bid_col = [c for c in df.columns if 'berdo id' in str(c).lower() and 'unnamed' not in str(c).lower()]
        if bid_col:
            out['building_id'] = 'boston_' + df[bid_col[0]].astype(str).str.strip()
        else:
            out['building_id'] = ['boston_row_' + str(year_str) + '_' + str(i) for i in range(len(df))]

        # Find property type column
        pt_col = [c for c in df.columns if 'largest property type' in str(c).lower() or
                  ('property type' in str(c).lower() and 'all' not in str(c).lower())]
        out['property_type'] = df[pt_col[0]].apply(map_property_type) if pt_col else np.nan

        # Year built — not in newer xlsx
        yb_col = [c for c in df.columns if 'year built' in str(c).lower()]
        out['year_built'] = to_numeric_safe(df[yb_col[0]]) if yb_col else np.nan

        # GFA
        gfa_col = [c for c in df.columns if 'reported gross floor area' in str(c).lower() or
                   'gross area' in str(c).lower()]
        out['gross_floor_area_sqft'] = to_numeric_safe(df[gfa_col[0]]) if gfa_col else np.nan

        # Site EUI
        eui_col = [c for c in df.columns if 'site eui' in str(c).lower()]
        out['site_eui_kbtu_sqft'] = to_numeric_safe(df[eui_col[0]]) if eui_col else np.nan

        out['source_eui_kbtu_sqft'] = np.nan

        # Energy Star
        es_col = [c for c in df.columns if 'energy star score' in str(c).lower()]
        out['energy_star_score'] = to_numeric_safe(df[es_col[0]]) if es_col else np.nan

        # GHG — in kg for xlsx files, need to / 1000
        ghg_col = [c for c in df.columns if 'total ghg' in str(c).lower() or
                   'estimated total ghg' in str(c).lower()]
        if ghg_col:
            ghg_vals = to_numeric_safe(df[ghg_col[0]])
            # Check if units are kgCO2e (values typically > 1000) vs MTCO2e
            if 'kg' in str(ghg_col[0]).lower():
                out['total_ghg_emissions_mtco2e'] = ghg_vals / 1000.0
            else:
                out['total_ghg_emissions_mtco2e'] = ghg_vals
        else:
            out['total_ghg_emissions_mtco2e'] = np.nan

        # Electricity (kWh)
        elec_col = [c for c in df.columns if 'electricity usage' in str(c).lower() and 'kwh' in str(c).lower()]
        out['electricity_kwh'] = to_numeric_safe(df[elec_col[0]]) if elec_col else np.nan

        # Natural gas (kBtu)
        gas_col = [c for c in df.columns if 'natural gas usage' in str(c).lower() and 'kbtu' in str(c).lower()]
        out['natural_gas_kbtu'] = to_numeric_safe(df[gas_col[0]]) if gas_col else np.nan

        out['latitude'] = np.nan
        out['longitude'] = np.nan

        out['year'] = int(year_str)
        dfs.append(make_unified(out, city='boston'))

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Boston: {len(result)} rows, years {result['year'].min()}-{result['year'].max()}")
    return result


def process_denver():
    """Denver benchmarking."""
    print("Processing Denver...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'denver_benchmarking', 'denver_2024.csv'), low_memory=False)

    out = pd.DataFrame()
    out['building_id'] = 'denver_' + df['Building_ID'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['Reporting_Year'])
    out['property_type'] = df['Primary_Property_Type_EPA_Calculated'].apply(map_property_type)
    out['year_built'] = to_numeric_safe(df['Year_Built'])
    out['gross_floor_area_sqft'] = to_numeric_safe(df['Master_Sq_Ft'])

    eui_col = [c for c in df.columns if 'site_eui' in c.lower() and 'weather' not in c.lower()][0]
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df[eui_col])

    out['source_eui_kbtu_sqft'] = np.nan  # Denver doesn't have Source EUI

    es_col = [c for c in df.columns if 'energy_star_score' in c.lower()][0]
    out['energy_star_score'] = to_numeric_safe(df[es_col])

    ghg_col = [c for c in df.columns if 'total_ghg' in c.lower()][0]
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df[ghg_col])

    # Electricity in kWh
    elec_col = [c for c in df.columns if 'electricity_use' in c.lower() and 'grid' in c.lower() and 'kwh' in c.lower()]
    if not elec_col:
        elec_col = [c for c in df.columns if 'electricity_use' in c.lower() and 'kwh' in c.lower()]
    out['electricity_kwh'] = to_numeric_safe(df[elec_col[0]]) if elec_col else np.nan

    # Natural gas in kBtu
    gas_col = [c for c in df.columns if 'natural_gas_use' in c.lower() and 'kbtu' in c.lower()]
    out['natural_gas_kbtu'] = to_numeric_safe(df[gas_col[0]]) if gas_col else np.nan

    out['latitude'] = np.nan
    out['longitude'] = np.nan

    result = make_unified(out, city='denver')
    print(f"  Denver: {len(result)} rows")
    return result


def process_philadelphia():
    """Philadelphia benchmarking."""
    print("Processing Philadelphia...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'philadelphia_benchmarking', 'philly_2023_reported.csv'), low_memory=False)

    out = pd.DataFrame()
    out['building_id'] = 'philadelphia_' + df['philadelphia_building_id'].astype(str).str.strip()
    out['year'] = to_numeric_safe(df['data_year'])
    out['property_type'] = df['primary_prop_type_epa_calc'].apply(map_property_type)
    out['year_built'] = to_numeric_safe(df['year_built'])
    out['gross_floor_area_sqft'] = to_numeric_safe(df['total_floor_area_bld_pk_ft2'])
    out['site_eui_kbtu_sqft'] = to_numeric_safe(df['site_eui_kbtuft2'])
    out['source_eui_kbtu_sqft'] = to_numeric_safe(df['source_eui_kbtuft2'])
    out['energy_star_score'] = to_numeric_safe(df['energy_star_score'])
    out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df['total_ghg_emissions_mtco2e'])
    # Philadelphia electricity and gas are in kBtu
    elec_kbtu = to_numeric_safe(df['electric_use_kbtu'])
    out['electricity_kwh'] = elec_kbtu / 3.412
    out['natural_gas_kbtu'] = to_numeric_safe(df['natural_gas_use_kbtu'])
    out['latitude'] = to_numeric_safe(df['y_lat'])
    out['longitude'] = to_numeric_safe(df['x_lon'])

    result = make_unified(out, city='philadelphia')
    print(f"  Philadelphia: {len(result)} rows")
    return result


def process_portland():
    """Portland: 3 XLSX files (2019, 2023, 2024)."""
    print("Processing Portland...")
    dfs = []

    files = sorted(glob.glob(os.path.join(RAW_DIR, 'portland_benchmarking', '*.xlsx')))
    for f in files:
        year_str = os.path.basename(f).replace('portland_', '').replace('.xlsx', '')
        # Find the correct sheet
        xls = pd.ExcelFile(f)
        sheet = [s for s in xls.sheet_names if 'energy performance' in s.lower()][0]
        df = pd.read_excel(f, sheet_name=sheet)
        # Clean column names: strip whitespace and newlines
        df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]

        out = pd.DataFrame()

        # Building ID
        bid_col = [c for c in df.columns if 'building id' in c.lower()]
        if bid_col:
            out['building_id'] = 'portland_' + df[bid_col[0]].astype(str).str.strip()
        else:
            out['building_id'] = ['portland_row_' + str(year_str) + '_' + str(i) for i in range(len(df))]

        # Property type — find flexibly
        pt_col = [c for c in df.columns if 'primary property type' in c.lower()]
        out['property_type'] = df[pt_col[0]].apply(map_property_type) if pt_col else np.nan
        out['year'] = int(year_str)

        yb_col = [c for c in df.columns if c.lower().strip() == 'year built']
        out['year_built'] = to_numeric_safe(df[yb_col[0]]) if yb_col else np.nan

        gfa_col = [c for c in df.columns if 'floor area' in c.lower()][0]
        out['gross_floor_area_sqft'] = to_numeric_safe(df[gfa_col])

        eui_col = [c for c in df.columns if 'site eui' in c.lower() and 'weather' not in c.lower()][0]
        out['site_eui_kbtu_sqft'] = to_numeric_safe(df[eui_col])

        out['source_eui_kbtu_sqft'] = np.nan  # Portland doesn't have Source EUI

        es_col = [c for c in df.columns if 'energy star' in c.lower()][0]
        out['energy_star_score'] = to_numeric_safe(df[es_col])

        ghg_col = [c for c in df.columns if 'ghg' in c.lower()][0]
        out['total_ghg_emissions_mtco2e'] = to_numeric_safe(df[ghg_col])

        # Electricity in kWh
        elec_col = [c for c in df.columns if 'electricity' in c.lower() and 'kwh' in c.lower() and 'renewable' not in c.lower()]
        out['electricity_kwh'] = to_numeric_safe(df[elec_col[0]]) if elec_col else np.nan

        # Natural gas in therms -> kBtu
        gas_col = [c for c in df.columns if 'natural gas' in c.lower() and 'therms' in c.lower()]
        out['natural_gas_kbtu'] = to_numeric_safe(df[gas_col[0]]) * 100 if gas_col else np.nan

        out['latitude'] = np.nan
        out['longitude'] = np.nan

        dfs.append(make_unified(out, city='portland'))

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Portland: {len(result)} rows, years {result['year'].min()}-{result['year'].max()}")
    return result


def process_melbourne():
    """Australian BEEC/NABERS export with city inferred from suburb/state."""
    print("Processing Australian BEEC/NABERS data...")
    df = pd.read_excel(os.path.join(RAW_DIR, 'melbourne_cbd', 'cbd_beec_2026.xlsx'), sheet_name=0)

    out = pd.DataFrame()

    suburb_col = [c for c in df.columns if 'suburb/city' in c.lower()][0]
    state_col = [c for c in df.columns if 'state' in c.lower()][0]
    addr_col = [c for c in df.columns if c.lower().strip() == 'address / building'][0]
    street_col = [c for c in df.columns if 'street address line 1' in c.lower()][0]

    out['city'] = [
        normalize_aus_city(suburb, state, address)
        for suburb, state, address in zip(df[suburb_col], df[state_col], df[addr_col])
    ]

    # Building ID — use geocode when available, otherwise a cleaned street-address key.
    geo_col = [c for c in df.columns if 'geocode' in c.lower()]
    if geo_col:
        geocode = df[geo_col[0]].astype(str).str.strip()
        fallback = df[street_col].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True)
        out['building_id'] = np.where(
            geocode.notna() & (geocode != '') & (geocode.str.lower() != 'nan'),
            'aus_' + geocode,
            'aus_' + fallback,
        )
    else:
        fallback = df[street_col].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True)
        out['building_id'] = 'aus_' + fallback

    # Extract year from IssueDate or CertifiedDate.
    issue_col = [c for c in df.columns if 'issuedate' in c.lower()]
    if issue_col:
        dates = pd.to_datetime(df[issue_col[0]], errors='coerce')
        out['year'] = dates.dt.year
    else:
        cert_col = [c for c in df.columns if 'certified date' in c.lower()]
        if cert_col:
            dates = pd.to_datetime(df[cert_col[0]], errors='coerce')
            out['year'] = dates.dt.year
        else:
            out['year'] = np.nan

    out['property_type'] = np.nan
    out['year_built'] = np.nan

    # Building NLA in m² -> ft².
    nla_col = [c for c in df.columns if 'building nla' in c.lower()][0]
    nla_m2 = to_numeric_safe(df[nla_col])
    out['gross_floor_area_sqft'] = nla_m2 * 10.764

    # Annual consumption behaves like building-level annual energy use.
    # NABERS/BEEC export "AnnualConsumption" is reported in MJ (not kWh);
    # a prior version of this script treated it as kWh, which over-stated
    # AU site EUI by ~3.6×. Unit reference: NABERS Rules for Collecting
    # and Using Data, and validated against Annual Emission Intensity
    # (~1.0 kgCO2/kWh Victorian grid → energy intensity consistent with
    # kWh-equivalent of the MJ column).
    cons_col = [c for c in df.columns if 'annualconsumption' in c.lower()][0]
    annual_consumption_mj = to_numeric_safe(df[cons_col])
    # NABERS export does not split electricity vs. gas — leave the
    # electricity-only column empty rather than mis-populate it.
    out['electricity_kwh'] = np.nan
    out['natural_gas_kbtu'] = np.nan

    # Compute site EUI from annual consumption (MJ) and building area (sqft).
    # 1 MJ = 0.9478 kBTU.
    out['site_eui_kbtu_sqft'] = (annual_consumption_mj * 0.9478) / out['gross_floor_area_sqft']
    out['source_eui_kbtu_sqft'] = np.nan

    # Keep NABERS stars in the score slot; downstream users should note the scale differs from ENERGY STAR.
    star_col = [c for c in df.columns if 'star rating' in c.lower()][0]
    out['energy_star_score'] = to_numeric_safe(df[star_col])

    emis_col = [c for c in df.columns if 'annual emissions' in c.lower() and 'intensity' not in c.lower()][0]
    ghg_raw = to_numeric_safe(df[emis_col])
    out['total_ghg_emissions_mtco2e'] = ghg_raw / 1000.0

    lat_col = [c for c in df.columns if 'latitude' in c.lower()][0]
    lon_col = [c for c in df.columns if 'longitude' in c.lower()][0]
    out['latitude'] = to_numeric_safe(df[lat_col])
    out['longitude'] = to_numeric_safe(df[lon_col])

    result = make_unified(out)
    print(f"  Australian BEEC rows: {len(result)} across {result['city'].nunique()} cities")
    return result


def process_singapore():
    """Singapore BCA: 2 CSV + 1 XLSX. Wide format -> melt.
    GHG estimated using electricity emission factor 0.44876 kgCO2/kWh."""
    print("Processing Singapore...")
    SGP_EMISSION_FACTOR = 0.44876  # kgCO2/kWh (from Paper #10)

    dfs = []

    # 2020 CSV (most complete, has 2017-2020 EUI)
    df = pd.read_csv(os.path.join(RAW_DIR, 'singapore_bca',
                                   'Listing of Building Energy Performance Data 2020.csv'))

    # EUI columns are year names: 2017, 2018, 2019, 2020
    year_cols = [c for c in df.columns if c.strip().isdigit()]

    # Parse GFA (has commas)
    gfa_raw = df['grossfloorarea'].astype(str).str.replace(',', '').str.strip()
    gfa_m2 = pd.to_numeric(gfa_raw, errors='coerce')

    # Building ID from building name (stable across years in wide-format data)
    sgp_ids = 'singapore_' + df['buildingname'].astype(str).str.strip().str.upper().str.replace(' ', '_')

    for yr_col in year_cols:
        yr = int(yr_col.strip())
        eui_kwh_m2 = to_numeric_safe(df[yr_col])

        row = pd.DataFrame()
        row['building_id'] = sgp_ids.values
        row['property_type'] = df['buildingtype'].apply(map_property_type)
        row['year'] = yr
        row['year_built'] = to_numeric_safe(df['yearobtainedtopcsc'])
        row['gross_floor_area_sqft'] = gfa_m2 * 10.764  # m² -> ft²
        # EUI: kWh/m²/yr -> kBtu/ft²/yr
        # 1 kWh = 3.412 kBtu, 1 m² = 10.764 ft²
        # kWh/m² * 3.412 / 10.764 = kBtu/ft²
        row['site_eui_kbtu_sqft'] = eui_kwh_m2 * 3.412 / 10.764
        row['source_eui_kbtu_sqft'] = np.nan
        row['energy_star_score'] = np.nan
        # GHG = EUI(kWh/m²) * GFA(m²) * emission_factor(kgCO2/kWh) / 1000 -> tCO2e
        row['total_ghg_emissions_mtco2e'] = eui_kwh_m2 * gfa_m2 * SGP_EMISSION_FACTOR / 1000.0
        row['electricity_kwh'] = eui_kwh_m2 * gfa_m2  # total electricity = EUI * GFA
        row['natural_gas_kbtu'] = np.nan  # Singapore is essentially all-electric
        row['latitude'] = np.nan
        row['longitude'] = np.nan

        dfs.append(make_unified(row, city='singapore'))

    # Commercial buildings CSV (2017-2018 EUI only)
    df2 = pd.read_csv(os.path.join(RAW_DIR, 'singapore_bca',
                                    'Listing of Building Energy Performance Data for Commercial Buildings.csv'))
    year_cols2 = [c for c in df2.columns if 'energyuse' in c.lower() or 'energyus' in c.lower()]
    gfa2_raw = df2['grossfloorarea'].astype(str).str.replace(',', '').str.strip()
    gfa2_m2 = pd.to_numeric(gfa2_raw, errors='coerce')

    sgp_ids2 = 'singapore_' + df2['buildingname'].astype(str).str.strip().str.upper().str.replace(' ', '_')

    for yr_col in year_cols2:
        # Extract year from column name like '2017energyuseintensity'
        yr = int(''.join(c for c in yr_col if c.isdigit())[:4])
        eui_kwh_m2 = to_numeric_safe(df2[yr_col])

        row = pd.DataFrame()
        row['building_id'] = sgp_ids2.values
        row['property_type'] = df2['buildingtype'].apply(map_property_type)
        row['year'] = yr
        row['year_built'] = np.nan
        row['gross_floor_area_sqft'] = gfa2_m2 * 10.764
        row['site_eui_kbtu_sqft'] = eui_kwh_m2 * 3.412 / 10.764
        row['source_eui_kbtu_sqft'] = np.nan
        row['energy_star_score'] = np.nan
        row['total_ghg_emissions_mtco2e'] = eui_kwh_m2 * gfa2_m2 * SGP_EMISSION_FACTOR / 1000.0
        row['electricity_kwh'] = eui_kwh_m2 * gfa2_m2
        row['natural_gas_kbtu'] = np.nan
        row['latitude'] = np.nan
        row['longitude'] = np.nan

        dfs.append(make_unified(row, city='singapore'))

    result = pd.concat(dfs, ignore_index=True)
    # Remove duplicate rows (buildings may appear in both files)
    result = result.drop_duplicates()
    print(f"  Singapore: {len(result)} rows")
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Building Dataset Standardization Pipeline")
    print("=" * 60)

    all_dfs = []

    processors = [
        process_nyc,
        process_chicago,
        process_seattle,
        process_dc,
        process_sf,
        process_la,
        process_boston,
        process_denver,
        process_philadelphia,
        process_portland,
        process_melbourne,
        process_singapore,
    ]

    for proc in processors:
        try:
            df = proc()
            all_dfs.append(df)
        except Exception as e:
            print(f"  ERROR in {proc.__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True)

    # Data cleaning: set unreasonable values to NaN
    # Negative EUI -> NaN
    neg_eui = combined['site_eui_kbtu_sqft'] < 0
    print(f"Setting {neg_eui.sum()} negative Site EUI values to NaN")
    combined.loc[neg_eui, 'site_eui_kbtu_sqft'] = np.nan

    neg_seui = combined['source_eui_kbtu_sqft'] < 0
    print(f"Setting {neg_seui.sum()} negative Source EUI values to NaN")
    combined.loc[neg_seui, 'source_eui_kbtu_sqft'] = np.nan

    # Extreme EUI (>1000 kBtu/ft² is very unusual — top 0.01% threshold)
    extreme_eui = combined['site_eui_kbtu_sqft'] > 1000
    print(f"Setting {extreme_eui.sum()} extreme Site EUI (>1000) values to NaN")
    combined.loc[extreme_eui, 'site_eui_kbtu_sqft'] = np.nan

    # Negative GHG -> NaN
    neg_ghg = combined['total_ghg_emissions_mtco2e'] < 0
    print(f"Setting {neg_ghg.sum()} negative GHG values to NaN")
    combined.loc[neg_ghg, 'total_ghg_emissions_mtco2e'] = np.nan

    # Extreme GHG (>100,000 tCO2e for a single building is unrealistic)
    extreme_ghg = combined['total_ghg_emissions_mtco2e'] > 100000
    print(f"Setting {extreme_ghg.sum()} extreme GHG (>100K) values to NaN")
    combined.loc[extreme_ghg, 'total_ghg_emissions_mtco2e'] = np.nan

    # Year built: 0 or unreasonable -> NaN
    bad_yb = (combined['year_built'] < 1800) | (combined['year_built'] > 2026)
    print(f"Setting {bad_yb.sum()} unreasonable Year Built values to NaN")
    combined.loc[bad_yb, 'year_built'] = np.nan

    # Energy Star Score = 0 likely means missing, not actual score
    es_zero = combined['energy_star_score'] == 0
    print(f"Setting {es_zero.sum()} Energy Star Score = 0 to NaN")
    combined.loc[es_zero, 'energy_star_score'] = np.nan

    # Remove rows with non-positive floor area
    before = len(combined)
    combined = combined[
        (combined['gross_floor_area_sqft'].isna()) |
        (combined['gross_floor_area_sqft'] > 0)
    ].reset_index(drop=True)
    print(f"Removed {before - len(combined)} rows with non-positive floor area")

    # Enforce one row per city/building/year after cleaning invalid values.
    combined, dup_rows = deduplicate_building_years(combined)
    print(f"Collapsed {dup_rows} duplicate city+building_id+year rows")

    # Remove rows where both GHG and EUI are missing (no useful target)
    has_ghg = combined['total_ghg_emissions_mtco2e'].notna()
    has_eui = combined['site_eui_kbtu_sqft'].notna()
    before = len(combined)
    combined = combined[has_ghg | has_eui].reset_index(drop=True)
    print(f"Removed {before - len(combined)} rows with both GHG and EUI missing")

    # Save
    out_path = os.path.join(OUT_DIR, 'building_all.csv')
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total rows: {len(combined)}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary by city:")
    print("=" * 60)
    summary = combined.groupby('city').agg(
        rows=('city', 'count'),
        year_min=('year', 'min'),
        year_max=('year', 'max'),
        ghg_non_null=('total_ghg_emissions_mtco2e', lambda x: x.notna().sum()),
        eui_non_null=('site_eui_kbtu_sqft', lambda x: x.notna().sum()),
    )
    print(summary.to_string())

    print("\nMissing rate per column:")
    missing = combined.isnull().mean().round(4) * 100
    for col in UNIFIED_COLS:
        print(f"  {col:40s} {missing[col]:6.2f}%")

    print("\nProperty type distribution:")
    print(combined['property_type'].value_counts().head(15).to_string())

    print("\nAll-null columns by city:")
    for city in sorted(combined['city'].unique()):
        city_df = combined[combined['city'] == city]
        all_null = [c for c in UNIFIED_COLS if c != 'city' and city_df[c].notna().sum() == 0]
        if all_null:
            print(f"  {city:15s}: {all_null}")

    # Building ID statistics
    print("\n" + "=" * 60)
    print("Building ID statistics:")
    print("=" * 60)
    for city in sorted(combined['city'].unique()):
        city_df = combined[combined['city'] == city]
        n_rows = len(city_df)
        n_unique = city_df['building_id'].nunique()
        has_row_id = city_df['building_id'].str.contains('_row_').sum()
        years_per_bld = city_df.groupby('building_id')['year'].nunique()
        multi_year = (years_per_bld > 1).sum()
        print(f"  {city:15s}: {n_rows:6d} rows, {n_unique:5d} unique IDs, "
              f"{multi_year:5d} multi-year buildings ({multi_year/max(n_unique,1)*100:.1f}%), "
              f"avg {years_per_bld.mean():.1f} years/building"
              + (f" [{has_row_id} fallback IDs]" if has_row_id > 0 else ""))

    write_validation_reports(combined)


if __name__ == '__main__':
    main()
