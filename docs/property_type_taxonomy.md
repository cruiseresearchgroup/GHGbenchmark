# GHGbench T2 — Canonical Property Type Taxonomy

This table is the single source of truth for `property_type` labels in
[data/processed/building_all_aus_merged.csv](../data/processed/building_all_aus_merged.csv).
The raw disclosure files from each city use slightly different wording (and
SF uses very coarse self-reported categories); all raw values are funnelled
through [scripts/property_type_mapping.json](../scripts/property_type_mapping.json)
and the heuristic fallback in
[scripts/standardize_buildings.py](../scripts/standardize_buildings.py) into
one of 22 canonical categories.

## Canonical categories (n = 22)

| Category | Typical raw forms mapped here | Cities where populated |
|---|---|---|
| `Office` | "Office", "Medical Office", "Bank Branch" | nyc, la, seattle, dc, chicago, boston, portland |
| `Multifamily Housing` | "Multifamily Housing", "Multifamily", "Residential", "Mixed Residential", "Other - Lodging/Residential" | all US + singapore |
| `Retail` | "Retail Store", "Mall", "Shopping Center" (NOT bare "Commercial") | nyc, la, seattle, dc, chicago |
| `Hotel` | "Hotel", "Motel", "Lodging" | all US |
| `K-12 School` | "K-12 School", "School" | nyc, dc, chicago, seattle |
| `College/University` | "College/University", "University" | nyc, dc, chicago |
| `Hospital/Medical` | "Hospital", "Medical Office", "Clinic" | all US |
| `Warehouse/Distribution` | "Warehouse", "Distribution Center" | nyc, la, seattle, dc |
| `Industrial` | "Manufacturing/Industrial", "Industrial" | nyc, la |
| `Worship` | "Worship Facility", "Church", "Synagogue", "Mosque" | nyc, la, seattle, dc |
| `Senior Living` | "Senior Living Community", "Nursing Home" | most US |
| `Supermarket/Grocery` | "Supermarket/Grocery Store" | la, chicago, seattle |
| `Restaurant` | "Restaurant", "Food Service" | small |
| `Parking` | "Parking", "Garage" | la, dc |
| `Laboratory` | "Laboratory" | seattle, chicago |
| `Fitness Center` | "Fitness Center/Health Club" | small |
| `Library` | "Library" | dc |
| `Residence Hall` | "Residence Hall/Dormitory" | nyc, dc |
| `Mixed Use` | "Mixed Use Property", "RES/COMMERCIAL USE" | nyc, la, dc |
| `Self-Storage` | "Self-Storage Facility" | nyc, la |
| `Commercial (Mixed)` | "Commercial", "Commercial - Port Facility" | **sf only** — SF raw data uses one coarse "Commercial" bucket that covers offices, retail, mixed. Do not collapse this into `Retail`: doing so inflates SF's Retail bucket to ~82% and misleads any model that splits offices vs. retail. |
| `Other` | "Other", "Mixed Use - Commercial", residuals | all |

## Australia (15 metros) and Singapore

These cities have `property_type` set to **NaN**. The NABERS/BEEC export does
not carry a standard building-use taxonomy aligned with ENERGY STAR PM, and
attempting to back-fill one from street address or NLA would be noise.
Downstream models in the `core_all_cities` feature set treat `property_type`
as an optional feature and these rows are handled by `us_core` / `us_metadata`
slices that restrict the city set to the six US cities with high coverage.

## History of fixes

- **2026-04-13** — Fixed a latent bug where SF's raw `"Commercial"` string was
  exact-matched to `"Retail"` via [scripts/property_type_mapping.json](../scripts/property_type_mapping.json),
  causing SF's property_type distribution to read as 82% Retail. Added a new
  `Commercial (Mixed)` category and a dedicated partial-match rule in
  [scripts/standardize_buildings.py::map_property_type](../scripts/standardize_buildings.py)
  so that the bare-`commercial` substring no longer leaks into `Retail`.
