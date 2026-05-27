"""Generate fig9_t2_cross_city.pdf — T2 City-LOCO per-city R^2 dot plot.

24 held-out cities × 3 tree models (RF, LGBM, XGB), sorted by RandomForest R^2.
Country colors. Reference lines at 0 and 0.411 (pooled grouped-building RF).
Off-scale points flagged with arrow markers and text labels.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "clean_building" / "task_c1_3seeds_raw.csv"
OUT = ROOT / "neurips_2026" / "figures" / "fig9_t2_cross_city.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

US = {"boston","chicago","dc","denver","la","nyc","philadelphia","portland","seattle","sf"}
AU = {"adelaide","brisbane","cairns","canberra","darwin","gold_coast","hobart","melbourne",
      "newcastle","perth","sydney","townsville","wollongong"}
SG = {"singapore"}

COUNTRY_COLOR = {"US": "#1f77b4", "AU": "#e07b00", "SG": "#7e57c2"}
MODEL_MARKER = {"RandomForest": ("o", 36, "RF"),
                "LightGBM":     ("D", 30, "LGBM"),
                "XGBoost":      ("^", 36, "XGB")}

X_MIN, X_MAX = -1.5, 0.65
GROUPED_RF = 0.411

CITY_LABEL = {
    "nyc": "NYC", "la": "LA", "sf": "SF", "dc": "DC",
    "gold_coast": "Gold Coast",
}

def country_of(city):
    if city in US: return "US"
    if city in AU: return "AU"
    return "SG"

def pretty(city):
    if city in CITY_LABEL: return CITY_LABEL[city]
    return city.replace("_", " ").title()


def main():
    df = pd.read_csv(SRC)
    df = df[(df.model.isin(MODEL_MARKER)) & (df.feature_set == "core_all_cities")]
    agg = df.groupby(["target_city", "model"])["r2"].agg(["mean", "std"]).reset_index()

    # sort cities by RF mean descending
    rf = agg[agg.model == "RandomForest"].set_index("target_city")["mean"]
    cities = rf.sort_values(ascending=False).index.tolist()
    y_pos = {c: i for i, c in enumerate(cities[::-1])}  # top of plot = best RF

    fig, ax = plt.subplots(figsize=(4.6, 5.0))

    # reference lines
    ax.axvline(0, color="#888", lw=0.7, zorder=1)
    ax.axvline(GROUPED_RF, color="#888", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(GROUPED_RF, len(cities) - 0.1, f" pooled grouped RF $\\approx$ {GROUPED_RF:.2f}",
            fontsize=6.8, color="#666", va="bottom", ha="left")

    # vertical jitter offsets per model so points don't overlap when close
    JITTER = {"RandomForest": 0.0, "LightGBM": -0.22, "XGBoost": +0.22}

    for _, row in agg.iterrows():
        c, m, mean = row["target_city"], row["model"], row["mean"]
        marker, size, _ = MODEL_MARKER[m]
        col = COUNTRY_COLOR[country_of(c)]
        y = y_pos[c] + JITTER[m]
        if mean < X_MIN:
            ax.scatter(X_MIN + 0.02, y, marker="<", s=24, color=col,
                       edgecolors="black", linewidths=0.4, zorder=3)
            ax.text(X_MIN + 0.07, y, f"{mean:.1f}", fontsize=5.6, color=col,
                    va="center", ha="left")
        else:
            ax.scatter(mean, y, marker=marker, s=size, color=col,
                       edgecolors="black", linewidths=0.4, zorder=3)

    # city y-tick labels colored by country
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([pretty(c) for c in cities[::-1]], fontsize=7.2,
                       fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), cities[::-1]):
        tick.set_color(COUNTRY_COLOR[country_of(c)])

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.7, len(cities) - 0.3)
    ax.set_xlabel(r"City-LOCO $R^2$ (3-seed mean)", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#bbb", zorder=0)
    ax.set_axisbelow(True)

    # combined legend (model shape + country color)
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker=mk, linestyle="", markersize=5.5,
               markerfacecolor="#444", markeredgecolor="black",
               markeredgewidth=0.4, label=lab)
        for mk, _, lab in MODEL_MARKER.values()
    ]
    country_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=6,
               markerfacecolor=COUNTRY_COLOR[c], markeredgecolor="black",
               markeredgewidth=0.3, label=c)
        for c in ("US", "AU", "SG")
    ]
    leg1 = ax.legend(handles=model_handles, loc="lower right",
                     fontsize=6.5, frameon=False, handletextpad=0.3,
                     borderaxespad=0.4, labelspacing=0.25,
                     bbox_to_anchor=(1.00, 0.00))
    ax.add_artist(leg1)
    ax.legend(handles=country_handles, loc="lower right",
              fontsize=6.5, frameon=False, handletextpad=0.3,
              borderaxespad=0.4, labelspacing=0.25,
              bbox_to_anchor=(1.00, 0.18))

    plt.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
