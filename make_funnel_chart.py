"""Chart of how the hard filters narrow the pool: python make_funnel_chart.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

report = json.load(open("results/SYNTHETIC_system_report.json"))
labels = [l for l, _ in report["funnel"]]
values = [v for _, v in report["funnel"]]
ink, muted, grid, color = "#1F2328", "#6B7280", "#E5E7EB", "#D9692F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 4.6))
y = list(range(len(values)))[::-1]
ax.barh(y, values, color=color, height=0.6)
for yi, v, prev in zip(y, values, [None] + values[:-1]):
    note = f"{v}" if prev is None else f"{v}   (−{prev - v})"
    ax.text(v + 6, yi, note, va="center", fontsize=11, color=ink, fontweight="bold")
ax.set_yticks(y, labels, color=ink)
ax.set_xlim(0, max(values) * 1.22)
ax.xaxis.grid(True, color=grid); ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False); ax.spines["bottom"].set_color(grid)
ax.tick_params(length=0, colors=muted)
ax.set_xlabel("Student pairs", color=muted)
fig.suptitle("How MeetKat's hard filters narrow 435 possible pairs", x=0.02, ha="left",
             fontsize=14, fontweight="bold", color=ink)
fig.text(0.02, 0.02, f"30 synthetic Hong Kong student profiles. Every student kept at least 3 compatible matches "
         f"(average {report['eligible_matches_per_student']['mean']:.1f}).", fontsize=9, color=muted)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig("results/SYNTHETIC_filter_funnel.png", dpi=200)
print("saved")
