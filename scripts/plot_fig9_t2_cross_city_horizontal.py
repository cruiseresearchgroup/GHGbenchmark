"""Regenerate fig9_t2_cross_city: rotate the original portrait scatter to landscape.

Preserves the original visual language:
  - 3 marker shapes (RF circle, LGBM diamond, XGB triangle)
  - country colours on city tick labels (US blue, AU orange, SG purple)
  - dashed line at pooled RF baseline 0.41
  - off-scale Hobart markers flagged at the floor

Only the orientation flips: cities now sit on the x-axis (sorted by RF
mean R² descending), R² on the y-axis. Marker faces are country-coloured;
edges thin black for legibility.

Source: results/clean_building/task_c1_3seeds_raw.csv
Output: neurips_2026/figures/fig9_t2_cross_city.pdf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

CSV = 'results/clean_building/task_c1_3seeds_raw.csv'
OUT = 'neurips_2026/figures/fig9_t2_cross_city.pdf'

US_CITIES = {'boston', 'chicago', 'denver', 'dc', 'la', 'nyc', 'philadelphia',
             'portland', 'seattle', 'sf'}
AU_CITIES = {'adelaide', 'brisbane', 'cairns', 'canberra', 'darwin',
             'gold_coast', 'hobart', 'melbourne', 'newcastle', 'perth',
             'sydney', 'townsville', 'wollongong'}

# Country colours (from previous portrait version)
COUNTRY_COLOR = {
    'US': '#1f77b4',  # blue
    'AU': '#ff7f0e',  # orange
    'SG': '#8c5dbe',  # purple
}
MODEL_MARKER = {
    'RandomForest': 'o',  # circle
    'LightGBM':     'D',  # diamond
    'XGBoost':      '^',  # triangle
}
DISPLAY_NAME = {
    'newcastle': 'Newcastle', 'cairns': 'Cairns', 'sf': 'SF', 'perth': 'Perth',
    'brisbane': 'Brisbane', 'portland': 'Portland', 'townsville': 'Townsville',
    'nyc': 'NYC', 'denver': 'Denver', 'darwin': 'Darwin',
    'gold_coast': 'Gold Coast', 'dc': 'DC', 'chicago': 'Chicago',
    'canberra': 'Canberra', 'sydney': 'Sydney', 'boston': 'Boston',
    'la': 'LA', 'adelaide': 'Adelaide', 'melbourne': 'Melbourne',
    'singapore': 'Singapore', 'philadelphia': 'Philadelphia',
    'wollongong': 'Wollongong', 'hobart': 'Hobart', 'seattle': 'Seattle',
}

POOLED_RF_BASELINE = 0.41

df = pd.read_csv(CSV)
df = df[(df['feature_set'] == 'core_all_cities')
        & (df['model'].isin(MODEL_MARKER.keys()))]

agg = (df.groupby(['model', 'target_city'])['r2']
         .agg(['mean', 'std']).reset_index())

# Sort cities by RF mean descending (best on left).
# Drop catastrophic cities (RF R² < -0.4): Hobart, Seattle. Their full per-city
# values appear in Appendix Table app_task_c1_city_r2.
rf = agg[agg['model'] == 'RandomForest'].set_index('target_city')['mean']
DROP_BELOW = -0.4
dropped_cities = rf[rf < DROP_BELOW].sort_values().index.tolist()
city_order = rf[rf >= DROP_BELOW].sort_values(ascending=False).index.tolist()
print(f"Dropped (in appendix): {dropped_cities}")
print(f"Shown ({len(city_order)} cities): {city_order}")


def country_of(city):
    if city in US_CITIES: return 'US'
    if city in AU_CITIES: return 'AU'
    return 'SG'


fig, ax = plt.subplots(figsize=(13, 3.2))

# Y-axis range tightened around the cities actually shown
Y_LOW, Y_HIGH = -0.45, 0.75

x_pos = np.arange(len(city_order))
x_offsets = {'RandomForest': -0.20, 'LightGBM': 0.0, 'XGBoost': 0.20}

offscale_records = []

for model in ['RandomForest', 'LightGBM', 'XGBoost']:
    sub = agg[agg['model'] == model].set_index('target_city')
    for i, city in enumerate(city_order):
        if city not in sub.index:
            continue
        m = float(sub.loc[city, 'mean'])
        c = COUNTRY_COLOR[country_of(city)]
        x = x_pos[i] + x_offsets[model]
        if m < Y_LOW:
            ax.scatter([x], [Y_LOW + 0.04], marker=MODEL_MARKER[model],
                       s=42, facecolors=c, edgecolors='black',
                       linewidths=0.5, zorder=3)
            ax.annotate('', xy=(x, Y_LOW - 0.02), xytext=(x, Y_LOW + 0.04),
                        arrowprops=dict(arrowstyle='-|>', color=c, lw=0.7))
            offscale_records.append((city, model, m))
        else:
            ax.scatter([x], [m], marker=MODEL_MARKER[model],
                       s=42, facecolors=c, edgecolors='black',
                       linewidths=0.5, zorder=3)

ax.axhline(POOLED_RF_BASELINE, ls='--', color='gray', lw=0.7)
ax.text(len(city_order) - 0.5, POOLED_RF_BASELINE + 0.03,
        f'pooled grouped RF $\\approx$ {POOLED_RF_BASELINE}',
        fontsize=7.5, color='gray', ha='right', va='bottom')
ax.axhline(0.0, color='black', lw=0.5, alpha=0.6)

# X-axis: country-coloured tick labels (rotated 45°)
ax.set_xticks(x_pos)
labels = []
for city in city_order:
    label = ax.get_xticklabels()  # placeholder
    pass
# Set tick labels with per-city colour
ax.set_xticklabels([DISPLAY_NAME.get(c, c.title()) for c in city_order],
                   rotation=45, ha='right', fontweight='bold', fontsize=8)
for tick_lbl, city in zip(ax.get_xticklabels(), city_order):
    tick_lbl.set_color(COUNTRY_COLOR[country_of(city)])

ax.set_ylabel('City-LOCO $R^2$ (3-seed mean)', fontsize=9, fontweight='bold')
ax.set_ylim(Y_LOW, Y_HIGH)
ax.set_yticks(np.arange(-0.4, 0.71, 0.2))
ax.grid(axis='y', alpha=0.25, lw=0.4, ls=':')
ax.set_axisbelow(True)

# Combined legend: country (color) and model (marker)
country_handles = [Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=COUNTRY_COLOR[c], markersize=7,
                          markeredgecolor='black', markeredgewidth=0.4,
                          label=c, linestyle='None')
                   for c in ['US', 'AU', 'SG']]
model_handles = [Line2D([0], [0], marker=MODEL_MARKER[m], color='w',
                        markerfacecolor='gray', markersize=6.5,
                        markeredgecolor='black', markeredgewidth=0.4,
                        label={'RandomForest': 'RF',
                               'LightGBM': 'LGBM',
                               'XGBoost': 'XGB'}[m],
                        linestyle='None')
                 for m in ['RandomForest', 'LightGBM', 'XGBoost']]
leg1 = ax.legend(handles=country_handles, loc='upper right',
                 bbox_to_anchor=(1.0, 1.0), fontsize=7.5, ncol=3,
                 columnspacing=0.7, handletextpad=0.3, frameon=False,
                 title='Country', title_fontsize=7.5)
ax.add_artist(leg1)
ax.legend(handles=model_handles, loc='lower right',
          bbox_to_anchor=(1.0, 0.0), fontsize=7.5, ncol=3,
          columnspacing=0.7, handletextpad=0.3, frameon=False,
          title='Model', title_fontsize=7.5)

# Note about omitted cities (catastrophic outliers in appendix table)
if dropped_cities:
    omitted = ", ".join(DISPLAY_NAME.get(c, c.title()) for c in dropped_cities)
    ax.text(0.01, 0.02,
            f"Omitted from this figure (RF $R^2 < {DROP_BELOW}$): {omitted}. "
            f"Full per-city table in Appendix.",
            transform=ax.transAxes, fontsize=6.8, ha='left', va='bottom',
            color='gray',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.75))

plt.tight_layout(pad=0.3)
plt.savefig(OUT, bbox_inches='tight', pad_inches=0.05)
print(f"Saved: {OUT}")
print(f"Off-scale: {offscale_records}")
