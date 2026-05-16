from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONDITIONS = ["Overall", "Correct", "Incorrect"]
COLORS = {
    "Overall": "#111111",
    "Correct": "#D62728",
    "Incorrect": "#1F77B4",
}
MARKERS = {
    "Overall": "o",
    "Correct": "s",
    "Incorrect": "^",
}
METRICS = [
    ("ce_cum", "Distribution CE"),
    ("mse_cum", "Distribution MSE"),
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def find_source(root: Path) -> Path:
    candidates = [
        root
        / "exp2(generation_blind)"
        / "paper_plots"
        / "appendix"
        / "png"
        / "fig2-2_identity_condition_probability_side_raw.csv",
        root
        / "exp2(generation_blind)"
        / "paper_plots"
        / "main"
        / "png"
        / "fig2-2_identity_condition_probability_side_raw.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("fig2-2_identity_condition_probability_side_raw.csv not found")


def compute_delta(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, metric_label in METRICS:
        sub = raw[raw["metric"].eq(metric)]
        for condition in CONDITIONS:
            for window_end in sorted(sub["window_end"].unique()):
                markov = sub[
                    sub["side"].eq("Markov")
                    & sub["condition"].eq(condition)
                    & sub["window_end"].eq(window_end)
                ].iloc[0]
                nonmarkov = sub[
                    sub["side"].eq("Non-Markov")
                    & sub["condition"].eq(condition)
                    & sub["window_end"].eq(window_end)
                ].iloc[0]
                delta = (markov["mean"] - nonmarkov["mean"]) / nonmarkov["mean"] * 100.0
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": metric_label,
                        "condition": condition,
                        "window_end": int(window_end),
                        "markov_mean": markov["mean"],
                        "nonmarkov_mean": nonmarkov["mean"],
                        "delta_pct": delta,
                        "markov_n": int(markov["n"]),
                        "nonmarkov_n": int(nonmarkov["n"]),
                    }
                )
    return pd.DataFrame(rows)


def style_axis(ax) -> None:
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="y", color="#d5d5d5", linewidth=0.75)
    ax.grid(axis="x", color="#e2e2e2", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8b8b8")
    ax.spines["bottom"].set_color("#b8b8b8")


def plot_delta(delta: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharex=True)

    for ax, (metric, title) in zip(axes, METRICS):
        sub = delta[delta["metric"].eq(metric)]
        for condition in CONDITIONS:
            cond = sub[sub["condition"].eq(condition)].sort_values("window_end")
            x = cond["window_end"].to_numpy()
            y = cond["delta_pct"].to_numpy()
            ax.plot(
                x,
                y,
                color=COLORS[condition],
                marker=MARKERS[condition],
                markersize=4.8,
                linewidth=1.75,
                label=condition,
            )
            for xi, yi in zip(x, y):
                offset_y = 6 if yi >= 0 else -8
                va = "bottom" if yi >= 0 else "top"
                ax.annotate(
                    f"{yi:.1f}%",
                    xy=(xi, yi),
                    xytext=(0, offset_y),
                    textcoords="offset points",
                    color=COLORS[condition],
                    fontsize=6.9,
                    ha="center",
                    va=va,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                )

        ax.axhline(0, color="#666666", linestyle=(0, (3, 2)), linewidth=0.9)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel("Context Length")
        ax.set_xticks([100, 200, 500, 1000])
        style_axis(ax)

    axes[0].set_ylabel("Delta (%)")
    axes[0].set_ylim(-9, 21.5)
    axes[1].set_ylim(-20, 130)
    axes[1].set_ylabel("Delta (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.18, wspace=0.22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paper = root / "exp2(generation_blind)" / "paper_plots"
    source = find_source(root)
    raw = pd.read_csv(source)
    delta = compute_delta(raw)

    stem = "fig2_distribution_delta_plot"
    delta_path = paper / "main" / "png" / f"{stem}_raw.csv"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(delta_path, index=False)

    plot_delta(
        delta,
        paper / "main" / "png" / f"{stem}.png",
        paper / "main" / "pdf" / f"{stem}.pdf",
    )

    print(source)
    print(paper / "main" / "png" / f"{stem}.png")
    print(paper / "main" / "pdf" / f"{stem}.pdf")
    print(delta_path)


if __name__ == "__main__":
    main()
