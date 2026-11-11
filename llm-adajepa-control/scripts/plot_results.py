"""Optional figure regeneration; requires matplotlib, not needed for the demo."""
from audit import ROOT, load_verified


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, _, summary = load_verified()
    rows = summary["methods"]
    labels = ["Adaptive only", "Error trigger", "Joint / sync", "Joint / async", "Joint / rules"]
    values = [r["success_rate"] * 100 for r in rows]
    colors = ["#31475e", "#9baabc", "#bd6a59", "#227a9f", "#9baabc"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    fig.patch.set_facecolor("#fafbfd")
    ax.set_facecolor("#fafbfd")
    bars = ax.barh(labels, values, color=colors, height=.57)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Task success (%)")
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    for bar, row, value in zip(bars, rows, values):
        ax.text(value + 1.4, bar.get_y() + bar.get_height()/2,
                f"{value:.2f}%  ({row['successes']}/600)", va="center", fontsize=10)
    ax.set_title("When a VLM planner disrupts an adaptive controller",
                 loc="left", fontsize=16, weight="bold", pad=31)
    ax.text(0, 1.035, "Async recovers 33.33 percentage points over sync; adaptive-only remains strongest.",
            transform=ax.transAxes, fontsize=10, color="#415267")
    fig.text(.02, .018, "Historical audit: 5 methods × 200 scenes × 3 training seeds. "
             "Simulation results; no new GPU run in this release.",
             fontsize=9, color="#415267")
    fig.subplots_adjust(left=.17, right=.97, top=.80, bottom=.15)
    target = ROOT / "assets" / "success_rates.png"
    fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
