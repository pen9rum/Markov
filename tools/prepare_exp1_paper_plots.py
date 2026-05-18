import math
import shutil
from pathlib import Path

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


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(
    src_png: Path,
    src_pdf: Path,
    dst_png_dir: Path,
    dst_pdf_dir: Path,
    dst_stem: str | None = None,
) -> None:
    ensure_dir(dst_png_dir)
    ensure_dir(dst_pdf_dir)
    png_name = f"{dst_stem}.png" if dst_stem else src_png.name
    pdf_name = f"{dst_stem}.pdf" if dst_stem else src_pdf.name
    shutil.copy2(src_png, dst_png_dir / png_name)
    shutil.copy2(src_pdf, dst_pdf_dir / pdf_name)


def ci95(p: float, n: float) -> float:
    if pd.isna(p) or pd.isna(n) or n <= 0:
        return np.nan
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def style_axis(ax, ylim=None) -> None:
    ax.set_facecolor("#f6f7f8")
    ax.grid(axis="y", color="#d5d9dd", alpha=0.95, linestyle="-", linewidth=0.72)
    ax.grid(axis="x", color="#e3e6e8", alpha=0.8, linestyle="-", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8bdc2")
    ax.spines["bottom"].set_color("#b8bdc2")
    if ylim is not None:
        ax.set_ylim(*ylim)


def save_figure(fig, png_dir: Path, pdf_dir: Path, stem: str) -> None:
    ensure_dir(png_dir)
    ensure_dir(pdf_dir)
    fig.savefig(png_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{stem}.pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panel(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str | None = None) -> None:
    rounds = sorted(df["rounds"].dropna().unique())
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("rounds")
        if sub.empty:
            continue
        x = sub["rounds"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        n = sub["samples"].to_numpy(dtype=float)
        yerr = np.asarray([ci95(p, samples) for p, samples in zip(y, n)], dtype=float)
        lower = np.clip(y - yerr, 0.0, 1.0)
        upper = np.clip(y + yerr, 0.0, 1.0)
        color = MODEL_COLORS[model]
        ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, y, color=color, marker="o", markersize=3.8, linewidth=1.65)

    style_axis(ax, ylim=(0.0, 1.02))
    ax.set_title(title, loc="left", pad=6, color="#111111")
    ax.set_xticks(rounds)
    ax.set_xlabel("Context rounds")
    if ylabel:
        ax.set_ylabel(ylabel)


def make_markov_strict_2x2(df: pd.DataFrame, png_dir: Path, pdf_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    specs = [
        ("markov_acc_strict", "(a) Strict accuracy"),
        ("markov_precision_strict", "(b) Strict precision"),
        ("markov_recall_strict", "(c) Strict recall"),
        ("markov_f1_strict", "(d) Strict F1"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), specs):
        plot_metric_panel(ax, df, metric, title, ylabel="Score" if ax in axes[:, 0] else None)

    handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], lw=2.0, marker="o", markersize=3.8, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.18, wspace=0.16, hspace=0.34)
    save_figure(fig, png_dir, pdf_dir, "markov_strict_metrics_2x2")


def make_forecasting_3panel(df: pd.DataFrame, png_dir: Path, pdf_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharex=True)
    specs = [
        ("tv", "(a) Total variation distance", "Total variation distance"),
        ("brier", "(b) Brier score", "Brier score"),
        ("evloss", "(c) EV loss", "EV loss"),
    ]
    rounds = sorted(df["rounds"].dropna().unique())
    for ax, (metric, title, ylabel) in zip(axes, specs):
        for model in MODEL_ORDER:
            sub = df[df["model"] == model].sort_values("rounds")
            if sub.empty:
                continue
            ax.plot(
                sub["rounds"].to_numpy(dtype=float),
                sub[metric].to_numpy(dtype=float),
                color=MODEL_COLORS[model],
                marker="o",
                markersize=3.8,
                linewidth=1.65,
            )
        style_axis(ax)
        ax.set_title(title, loc="left", pad=6, color="#111111")
        ax.set_xticks(rounds)
        ax.set_xlabel("Context rounds")
        ax.set_ylabel(ylabel)

    handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], lw=2.0, marker="o", markersize=3.8, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.990, top=0.88, bottom=0.24, wspace=0.28)
    save_figure(fig, png_dir, pdf_dir, "forecasting_metrics_3panel")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp = root / "exp1(strategy)"
    plots = exp / "plots"
    paper = exp / "paper_plots"

    main_png = paper / "main" / "png"
    main_pdf = paper / "main" / "pdf"
    app_png = paper / "appendix" / "png"
    app_pdf = paper / "appendix" / "pdf"

    copy_pair(
        plots / "others" / "png" / "accuracy_8lines_markov_nonmarkov_95ci.png",
        plots / "others" / "pdf" / "accuracy_8lines_markov_nonmarkov_95ci.pdf",
        main_png,
        main_pdf,
    )

    for stem in ["ols_input_length_vs_accuracy", "ols_output_length_vs_accuracy"]:
        copy_pair(
            plots / "others" / "png" / f"{stem}.png",
            plots / "others" / "pdf" / f"{stem}.pdf",
            app_png,
            app_pdf,
        )

    copy_pair(
        plots / "confusion_identity" / "confusion_overall_all.png",
        plots / "confusion_identity" / "confusion_overall_all.pdf",
        paper / "appendix" / "confusion_identity" / "png",
        paper / "appendix" / "confusion_identity" / "pdf",
    )
    copy_pair(
        plots / "confusion_markov" / "confusion_overall_all.png",
        plots / "confusion_markov" / "confusion_overall_all.pdf",
        paper / "appendix" / "confusion_matrix" / "png",
        paper / "appendix" / "confusion_matrix" / "pdf",
        dst_stem="confusion_matrix_all",
    )

    df = pd.read_csv(exp / "metrics_export.csv")
    df = df[df["type"] == "overall"].copy()
    df["rounds"] = pd.to_numeric(df["rounds"])
    df["samples"] = pd.to_numeric(df["samples"])
    make_markov_strict_2x2(df, app_png, app_pdf)
    make_forecasting_3panel(df, app_png, app_pdf)
    copy_pair(
        plots / "others" / "png" / "accuracy_8lines_markov_nonmarkov_95ci_gap.png",
        plots / "others" / "pdf" / "accuracy_8lines_markov_nonmarkov_95ci_gap.pdf",
        app_png,
        app_pdf,
    )

    print(main_png / "accuracy_8lines_markov_nonmarkov_95ci.png")
    print(app_png / "ols_input_length_vs_accuracy.png")
    print(app_png / "ols_output_length_vs_accuracy.png")
    print(paper / "appendix" / "confusion_identity" / "png" / "confusion_overall_all.png")
    print(paper / "appendix" / "confusion_matrix" / "png" / "confusion_matrix_all.png")
    print(app_png / "markov_strict_metrics_2x2.png")
    print(app_png / "forecasting_metrics_3panel.png")
    print(app_png / "accuracy_8lines_markov_nonmarkov_95ci_gap.png")


if __name__ == "__main__":
    main()
