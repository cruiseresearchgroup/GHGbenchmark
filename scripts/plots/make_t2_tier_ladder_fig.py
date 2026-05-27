"""Generate fig10_t2_tier_ladder.pdf — T2 Task A feature-tier ladder.

Per-tier 3-seed mean grouped-building R^2 for TabPFN v2, RandomForest,
LightGBM, XGBoost across nine feature tiers organised in three scopes
(cross-country / US / AU). Background bands distinguish clean,
proxy-rich (EUI / NABERS) and direct-energy-proxy tiers.
"""

from pathlib import Path
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "neurips_2026" / "figures" / "fig10_t2_tier_ladder.pdf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

TIERS = [
    ("core_all_cities",              "Core",        "XC"),
    ("core_all_cities_climate_plus", "+climate",    "XC"),
    ("us_core",                      "Core",        "US"),
    ("us_metadata",                  "+metadata",   "US"),
    ("us_leaky_eui",                 "+EUI",        "US"),
    ("us_leaky_full",                "+energy",     "US"),
    ("au_core",                      "Core",        "AU"),
    ("au_eui",                       "+NABERS",     "AU"),
    ("au_full",                      "+energy",     "AU"),
]
PROXY_RICH = {"us_leaky_eui", "au_eui"}
DIRECT_ENERGY = {"us_leaky_full", "au_full"}

MODELS = [
    ("TabPFN",       "TabPFN v2",    "o", "#c43c4a"),
    ("RandomForest", "RF",           "D", "#1f77b4"),
    ("LightGBM",     "LightGBM",     "^", "#2ca02c"),
    ("XGBoost",      "XGBoost",      "s", "#ff7f0e"),
    ("MLP",          "MLP",          "v", "#7e57c2"),
]
SINGLE_SEED_MLP = {"us_core", "us_metadata", "us_leaky_eui", "us_leaky_full",
                   "au_core", "au_eui", "au_full"}


def load():
    tier_names = [t[0] for t in TIERS]
    tree_rows = []
    for t in tier_names:
        for path in glob.glob(str(ROOT / "results" / f"task_a_results_{t}.csv")):
            d = pd.read_csv(path)
            d["feature_set"] = t
            tree_rows.append(d)
    trees = pd.concat(tree_rows, ignore_index=True)
    trees = trees[(trees.task == "A2_pooled") & (trees.split_type == "grouped") &
                  (trees.model.isin(["RandomForest", "LightGBM", "XGBoost",
                                      "CityTypeMean"]))]
    tree_agg = (trees.groupby(["feature_set", "model"])["overall_r2"]
                .agg(["mean", "std"]).reset_index())

    tp = pd.read_csv(ROOT / "results" / "clean_building" / "task_a_tabpfn_3seeds_raw.csv")
    tp_agg = tp.groupby("feature_set")["r2"].agg(["mean", "std"]).reset_index()
    tp_agg["model"] = "TabPFN"

    # MLP: prefer 3-seed file for core_all_cities and climate_plus; single-seed
    # tier files for the remaining seven tiers.
    mlp_rows = []
    mlp3 = pd.read_csv(ROOT / "results" / "task_a_results_mlp_3seeds.csv")
    mlp3 = mlp3[(mlp3.task == "A2_pooled") & (mlp3.split_type == "grouped")]
    mlp3g = (mlp3.groupby("feature_set")["overall_r2"]
             .agg(["mean", "std"]).reset_index())
    mlp_rows.append(mlp3g)
    single_files = {
        "us_core":         "task_a_results_us_core_mlp.csv",
        "us_metadata":     "task_a_results_us_metadata_mlp.csv",
        "us_leaky_eui":    "task_a_results_us_leaky_eui_mlp.csv",
        "us_leaky_full":   "task_a_results_us_leaky_full_mlp_grouped_fix.csv",
        "au_core":         "task_a_results_au_core_mlp.csv",
        "au_eui":          "task_a_results_au_eui_mlp.csv",
        "au_full":         "task_a_results_au_full_mlp.csv",
    }
    for tier, fname in single_files.items():
        d = pd.read_csv(ROOT / "results" / fname)
        d = d[(d.task == "A2_pooled") & (d.split_type == "grouped")]
        if len(d):
            mlp_rows.append(pd.DataFrame({
                "feature_set": [tier],
                "mean": [d["overall_r2"].mean()],
                "std":  [np.nan],
            }))
    mlp_agg = pd.concat(mlp_rows, ignore_index=True)
    mlp_agg["model"] = "MLP"

    return pd.concat([tree_agg, tp_agg, mlp_agg], ignore_index=True)


def main():
    agg = load()

    # x positions: small gap between scope groups
    x_pos = []
    last_scope = None
    cur = 0.0
    for _, _, scope in TIERS:
        if last_scope is not None and scope != last_scope:
            cur += 0.6
        x_pos.append(cur)
        cur += 1.0
        last_scope = scope
    x_pos = np.array(x_pos)

    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    # Background tier-type bands
    for i, (tname, _, _) in enumerate(TIERS):
        if tname in DIRECT_ENERGY:
            ax.axvspan(x_pos[i] - 0.45, x_pos[i] + 0.45, color="#fde8c5",
                       alpha=0.85, zorder=0)
        elif tname in PROXY_RICH:
            ax.axvspan(x_pos[i] - 0.45, x_pos[i] + 0.45, color="#fdf3df",
                       alpha=0.85, zorder=0)

    # Scope group separators
    scope_starts = {}
    for i, (_, _, scope) in enumerate(TIERS):
        scope_starts.setdefault(scope, i)
    for scope, idx in list(scope_starts.items())[1:]:
        ax.axvline(x_pos[idx] - 0.8, color="#bbb", lw=0.6, zorder=0)

    # CityTypeMean as no-information floor reference (thin dashed grey)
    sub_floor = agg[agg.model == "CityTypeMean"].set_index("feature_set")
    floor_y = [sub_floor.loc[t[0], "mean"] if t[0] in sub_floor.index else np.nan
               for t in TIERS]
    ax.plot(x_pos, floor_y, color="#888", lw=0.9, ls=(0, (3, 2)),
            marker="x", ms=4, mew=0.7, label="CityTypeMean (floor)", zorder=2)

    # Plot one line per main model; MLP gets seed-aware handling
    for model, label, marker, color in MODELS:
        sub = agg[agg.model == model].set_index("feature_set")
        ys = [sub.loc[t[0], "mean"] if t[0] in sub.index else np.nan for t in TIERS]
        if model == "MLP":
            es = [sub.loc[t[0], "std"] if t[0] in sub.index else np.nan
                  for t in TIERS]
            ax.plot(x_pos, ys, color=color, lw=1.0, alpha=0.7, zorder=3,
                    label=label)
            for i, (tname, _, _) in enumerate(TIERS):
                if tname in SINGLE_SEED_MLP:
                    ax.scatter(x_pos[i], ys[i], marker=marker, s=30,
                               facecolors="white", edgecolors=color,
                               linewidths=1.0, zorder=4)
                else:
                    ax.errorbar(x_pos[i], ys[i], yerr=es[i], fmt=marker, ms=5.5,
                                color=color, mec="black", mew=0.4,
                                capsize=2.0, elinewidth=0.6, zorder=4)
        else:
            es = [sub.loc[t[0], "std"] if t[0] in sub.index else np.nan
                  for t in TIERS]
            ax.errorbar(x_pos, ys, yerr=es, fmt=marker, ms=5.5, color=color,
                        mec="black", mew=0.4, lw=1.0, capsize=2.0,
                        elinewidth=0.6, label=label, zorder=4)
            ax.plot(x_pos, ys, color=color, lw=1.0, alpha=0.7, zorder=3)

    # x-axis tick labels (tier-specific) + scope group labels above
    ax.set_xticks(x_pos)
    ax.set_xticklabels([t[1] for t in TIERS], fontsize=7.5, fontweight="bold",
                       rotation=0)
    # scope band labels
    scope_label_y = 1.02
    scope_groups = {}
    for i, (_, _, scope) in enumerate(TIERS):
        scope_groups.setdefault(scope, []).append(x_pos[i])
    scope_full = {"XC": "Cross-country", "US": "US-only", "AU": "AU-only"}
    for scope, xs in scope_groups.items():
        x_mid = (min(xs) + max(xs)) / 2
        ax.text(x_mid, scope_label_y, scope_full[scope], transform=ax.get_xaxis_transform(),
                fontsize=8.5, fontweight="bold", ha="center", va="bottom")

    ax.set_ylim(-0.55, 0.95)
    ax.set_ylabel(r"Grouped-building $R^2$ (3-seed mean)",
                  fontsize=8, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#bbb", zorder=1)
    ax.set_axisbelow(True)

    # background-band legend (text annotations on right edge)
    ax.text(x_pos[-1] + 0.55, 0.74, "proxy-rich (EUI/NABERS)",
            fontsize=6.5, color="#946d2c", rotation=90, va="center", ha="left")
    ax.text(x_pos[-1] + 0.85, 0.74, "+direct energy",
            fontsize=6.5, color="#7c4f15", rotation=90, va="center", ha="left")

    ax.legend(loc="upper left", fontsize=7, frameon=False,
              handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3,
              ncol=6, columnspacing=1.0, bbox_to_anchor=(0.0, -0.20))

    plt.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
