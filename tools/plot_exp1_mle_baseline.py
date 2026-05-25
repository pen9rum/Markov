from pathlib import Path
import math
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODEL_ORDER = ["deepseek-chat", "deepseek-reasoner", "gpt-5", "gpt-5-mini"]
MODEL_COLORS = {
    "deepseek-chat": "#1f77b4",
    "deepseek-reasoner": "#ff7f0e",
    "gpt-5": "#2ca02c",
    "gpt-5-mini": "#d62728",
}
MODEL_LABELS = {
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-reasoner": "DeepSeek Reasoner",
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 mini",
}
MARKOV_IDS = {"X", "Y", "Z"}
EXPORT_DPI = 400


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "legend.fontsize": 8.6,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mirror_to_appendix(src_png: Path, src_pdf: Path, appendix_png: Path, appendix_pdf: Path) -> None:
    ensure_dir(appendix_png)
    ensure_dir(appendix_pdf)
    shutil.copy2(src_png, appendix_png / src_png.name)
    shutil.copy2(src_pdf, appendix_pdf / src_pdf.name)


def save_fig(fig, png_dir: Path, pdf_dir: Path, stem: str) -> tuple[Path, Path]:
    ensure_dir(png_dir)
    ensure_dir(pdf_dir)
    png = png_dir / f"{stem}.png"
    pdf = pdf_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def binomial_ci_95(p: float, n: float) -> float:
    if pd.isna(p) or pd.isna(n) or n <= 0:
        return np.nan
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def mean_ci(vals):
    arr = np.asarray([v for v in vals if pd.notna(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    ci = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean, ci


def style_axis(ax):
    ax.set_facecolor("#f6f7f8")
    ax.grid(axis="y", color="#d5d9dd", alpha=0.95, linestyle="-", linewidth=0.72)
    ax.grid(axis="x", color="#e3e6e8", alpha=0.65, linestyle="-", linewidth=0.50)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8bdc2")
    ax.spines["bottom"].set_color("#b8bdc2")


def parse_top5_probs(value: str) -> list[float]:
    return [max(float(x), 1e-300) for x in str(value).split("|") if x != ""]


def player_probability_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in detail.iterrows():
        for player in ("p1", "p2"):
            gt = row["gt_player1"] if player == "p1" else row["gt_player2"]
            pred = row["pred_player1"] if player == "p1" else row["pred_player2"]
            probs = parse_top5_probs(row[f"{player}_top5_prob"])
            ids = str(row[f"{player}_top5_identities"]).split("|")
            true_class = "Markov" if gt in MARKOV_IDS else "Non-Markov"
            for rank, (candidate, prob) in enumerate(zip(ids, probs), start=1):
                rows.append(
                    {
                        "source_model": row["source_model"],
                        "rounds": int(row["rounds"]),
                        "type": row["type"],
                        "player": player,
                        "gt_identity": gt,
                        "pred_identity": pred,
                        "true_class": true_class,
                        "rank": rank,
                        "candidate": candidate,
                        "probability": prob,
                    }
                )
    return pd.DataFrame(rows)


def plot_top5_probabilities(detail_csv: Path, png_dir: Path, pdf_dir: Path, appendix_png: Path, appendix_pdf: Path) -> None:
    detail = pd.read_csv(detail_csv)
    probs = player_probability_rows(detail)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.8, 4.2), sharex="col",
        gridspec_kw={"height_ratios": [0.82, 1.65], "hspace": 0.05, "wspace": 0.24},
    )
    specs = [
        ("Non-Markov", "#4477AA"),
        ("Markov", "#EE6677"),
    ]
    ranks = np.arange(1, 6)

    for col, (true_class, color) in enumerate(specs):
        ax_top = axes[0, col]
        ax_tail = axes[1, col]
        sub = probs[probs["true_class"] == true_class]
        means, cis = [], []
        for rank in ranks:
            mean, ci = mean_ci(sub[sub["rank"] == rank]["probability"])
            means.append(mean)
            cis.append(ci)
        means = np.asarray(means)
        cis = np.asarray(cis)
        lower = np.maximum(means - cis, 1e-300)
        upper = np.minimum(means + cis, 1.0)

        # Upper segment: show that rank-1 is essentially at ceiling.
        ax_top.errorbar([1], [means[0]], yerr=[[max(cis[0], 0)], [max(cis[0], 0)]],
                        color="black", capsize=2.2, linewidth=0.8, fmt="none", zorder=3)
        ax_top.plot([1], [means[0]], color=color, marker="o", markersize=5.5, zorder=4)
        ax_top.annotate(f"{means[0]:.3f}", xy=(1, means[0]), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.6, color="#111111")
        ax_top.set_ylim(0.95, 1.01)
        ax_top.set_xlim(0.45, 5.55)
        ax_top.set_title(f"{true_class} true players")
        style_axis(ax_top)
        ax_top.grid(axis="x", visible=False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

        # Lower segment: shared log scale across panels, so left-side ranks 2-5
        # remain visible while the much smaller Markov tail stays near the floor.
        tail_ranks = ranks[1:]
        tail_means = np.maximum(means[1:], 1e-300)
        tail_lower = np.maximum(lower[1:], 1e-300)
        tail_upper = np.minimum(upper[1:], 1.0)
        ymin = 1e-35
        ymax = 1e-1
        ax_tail.fill_between(tail_ranks, tail_lower, tail_upper, color=color, alpha=0.16, linewidth=0)
        ax_tail.plot(tail_ranks, tail_means, color=color, marker="o", linewidth=2.1, markersize=5.0)
        for rank, value in zip(tail_ranks, means[1:]):
            plot_value = max(value, ymin)
            ax_tail.annotate(f"{value:.1e}", xy=(rank, plot_value), xytext=(0, 9),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=7.4, color="#111111")
        ax_tail.set_yscale("log")
        ax_tail.set_ylim(ymin, ymax)
        ax_tail.set_xlim(0.45, 5.55)
        ax_tail.set_xlabel("Candidate rank")
        ax_tail.set_xticks(ranks)
        style_axis(ax_tail)
        ax_tail.spines["top"].set_visible(False)

        # Break marks between the ceiling band and log-tail band.
        kwargs = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=6,
                      linestyle="none", color="0.35", mec="0.35", mew=0.8, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_tail.plot([0, 1], [1, 1], transform=ax_tail.transAxes, **kwargs)

    axes[0, 0].set_ylabel("Top-1 posterior")
    axes[1, 0].set_ylabel("Ranks 2-5 posterior")
    fig.tight_layout(rect=[0, 0.02, 1, 1], pad=0.8)
    png, pdf = save_fig(fig, png_dir, pdf_dir, "mle_top5_probabilities_markov_nonmarkov")
    mirror_to_appendix(png, pdf, appendix_png, appendix_pdf)
    print(png)
    print(pdf)


def plot_mle_vs_llm(metrics_csv: Path, baseline_summary_csv: Path, png_dir: Path, pdf_dir: Path, appendix_png: Path, appendix_pdf: Path) -> None:
    llm = pd.read_csv(metrics_csv)
    llm = llm[(llm["type"] == "overall") & (llm["model"].isin(MODEL_ORDER))].copy()
    llm["rounds"] = pd.to_numeric(llm["rounds"])
    llm["samples"] = pd.to_numeric(llm["samples"])

    baseline = pd.read_csv(baseline_summary_csv)
    baseline = baseline[(baseline["source_model"] == "all_main_sources") & (baseline["type"] == "overall")].copy()
    baseline["rounds"] = pd.to_numeric(baseline["rounds"])
    baseline["samples"] = pd.to_numeric(baseline["samples"])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), sharey=True)
    panel_specs = [
        ("nonmarkov_acc", "nonmarkov_player_accuracy", "(a) Non-Markov players"),
        ("markov_recall_strict", "markov_player_accuracy", "(b) Markov players"),
    ]

    for ax, (llm_col, mle_col, title) in zip(axes, panel_specs):
        for model in MODEL_ORDER:
            sub = llm[llm["model"] == model].sort_values("rounds")
            if sub.empty:
                continue
            x = sub["rounds"].to_numpy(dtype=float)
            y = sub[llm_col].to_numpy(dtype=float)
            n = sub["samples"].to_numpy(dtype=float)
            ci = np.asarray([binomial_ci_95(p, samples) for p, samples in zip(y, n)], dtype=float)
            ax.fill_between(x, np.clip(y - ci, 0, 1), np.clip(y + ci, 0, 1),
                            color=MODEL_COLORS[model], alpha=0.11, linewidth=0)
            ax.plot(x, y, color=MODEL_COLORS[model], marker="o", linewidth=1.65,
                    markersize=3.8, label=MODEL_LABELS[model])

        mle = baseline.sort_values("rounds")
        ax.plot(
            mle["rounds"].to_numpy(dtype=float),
            mle[mle_col].to_numpy(dtype=float),
            color="black",
            linestyle=(0, (4, 2)),
            marker="s",
            linewidth=1.85,
            markersize=4.0,
            label="MLE baseline",
        )
        ax.set_title(title, loc="left", pad=7)
        ax.set_xlabel("Context rounds")
        ax.set_xticks(sorted(llm["rounds"].unique()))
        ax.set_ylim(0, 1.04)
        style_axis(ax)

    axes[0].set_ylabel("Identification accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    # Add the MLE handle from the second axis if it is not already present.
    h2, l2 = axes[1].get_legend_handles_labels()
    for h, label in zip(h2, l2):
        if label not in labels:
            handles.append(h)
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, handlelength=1.8, columnspacing=0.95)
    fig.subplots_adjust(left=0.080, right=0.990, top=0.90, bottom=0.25, wspace=0.16)
    png, pdf = save_fig(fig, png_dir, pdf_dir, "mle_vs_llm_identification_accuracy")
    mirror_to_appendix(png, pdf, appendix_png, appendix_pdf)
    print(png)
    print(pdf)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "exp1(strategy)"
    png_dir = exp / "plots" / "others" / "png"
    pdf_dir = exp / "plots" / "others" / "pdf"
    appendix_png = exp / "paper_plots" / "appendix" / "png"
    appendix_pdf = exp / "paper_plots" / "appendix" / "pdf"

    plot_top5_probabilities(
        exp / "analysis_results" / "mle_identification_baseline_detail.csv",
        png_dir,
        pdf_dir,
        appendix_png,
        appendix_pdf,
    )
    plot_mle_vs_llm(
        exp / "metrics_export.csv",
        exp / "analysis_results" / "mle_identification_baseline_summary.csv",
        png_dir,
        pdf_dir,
        appendix_png,
        appendix_pdf,
    )


if __name__ == "__main__":
    main()
