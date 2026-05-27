"""Generate fig8_t1_attribution.pdf (T1-A LightGBM permutation + SHAP, top 6).

Numbers mirror tables/app_t1_feature_attribution.tex so the wrapfigure in §5.2
stays consistent with the appendix table. Re-run only if the underlying
attribution numbers change.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

OUT = Path(__file__).resolve().parents[2] / "neurips_2026" / "figures" / "fig8_t1_attribution.pdf"

features = ["Revenue", "ExioML factor", "EBITDA", "Employees", "Market cap", "GICS Fin.~Svc."]
perm = np.array([0.652, 0.266, 0.207, 0.178, 0.161, 0.035])
perm_err = np.array([0.016, 0.013, 0.008, 0.011, 0.007, 0.003])
shap = np.array([0.477, 0.298, 0.203, 0.124, 0.110, 0.070])

y = np.arange(len(features))[::-1]

fig, ax = plt.subplots(figsize=(3.4, 2.1))
bars = ax.barh(y, perm, xerr=perm_err, color="#3b6ea8", height=0.6,
               error_kw={"elinewidth": 0.6, "capsize": 1.8, "ecolor": "#222"})
for yi, p, s in zip(y, perm, shap):
    ax.text(p + 0.018, yi, f"{p:.2f}", va="center", ha="left", fontsize=7, color="#222")

ax.set_yticks(y)
ax.set_yticklabels(features, fontsize=7.5, fontweight="bold")
ax.set_xlabel(r"Permutation $\Delta R^2_{\log}$", fontsize=7.5, fontweight="bold")
ax.tick_params(axis="x", labelsize=7)
ax.set_xlim(0, 0.78)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#bbb", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.2)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT}")
