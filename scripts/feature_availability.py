"""Build a city x feature availability matrix + heatmap.

Reads the canonical building table directly — geocoded lat/lon for LA and
Denver is already baked in by scripts/merge_aus_regions.py, so no special
load pipeline is needed.

Outputs:
  docs/feature_availability.csv     — non-null % per (city, feature)
  docs/feature_availability.png     — heatmap visualisation
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BUILDING_CSV = "data/processed/building_all_aus_merged.csv"
OUT_CSV = "docs/feature_availability.csv"
OUT_PNG = "docs/feature_availability.png"

FEATURES = [
    "property_type",
    "year_built",
    "gross_floor_area_sqft",
    "site_eui_kbtu_sqft",
    "source_eui_kbtu_sqft",
    "energy_star_score",
    "electricity_kwh",
    "natural_gas_kbtu",
    "latitude",
    "longitude",
    "total_ghg_emissions_mtco2e",
]

df = pd.read_csv(BUILDING_CSV, low_memory=False)

rows = []
for city, g in df.groupby("city"):
    n = len(g)
    rec = {"city": city, "n_rows": n, "n_buildings": g["building_id"].nunique()}
    for f in FEATURES:
        rec[f] = round(g[f].notna().mean() * 100, 1) if f in g.columns else np.nan
    rows.append(rec)

avail = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
os.makedirs("docs", exist_ok=True)
avail.to_csv(OUT_CSV, index=False)
print(avail.to_string(index=False))

# Heatmap (cities sorted by row count, features in fixed order)
mat = avail.set_index("city")[FEATURES].astype(float)
fig_h = max(6, 0.28 * len(mat))
fig, ax = plt.subplots(figsize=(12, fig_h))
im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(mat)))
ax.set_yticklabels(
    [f"{c}  (n={int(avail.loc[i,'n_rows']):,})" for i, c in enumerate(mat.index)],
    fontsize=9,
)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        if np.isnan(v):
            txt = "—"
        else:
            txt = f"{v:.0f}"
        color = "white" if (not np.isnan(v) and (v < 25 or v > 85)) else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Non-null % of rows", fontsize=9)
ax.set_title(
    "GHGbench T2 — Feature Availability per City\n"
    "(green = feature densely populated, red = mostly missing)",
    fontsize=12, pad=12,
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"\nwrote {OUT_CSV}")
print(f"wrote {OUT_PNG}")
