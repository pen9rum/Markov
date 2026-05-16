import argparse
import math
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
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.6,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def binomial_ci_95(p: float, n: float) -> float:
    if pd.isna(p) or pd.isna(n) or n <= 0:
        return np.nan
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def diff_ci_95(p_markov: float, p_nonmarkov: float, n: float) -> float:
    if pd.isna(p_markov) or pd.isna(p_nonmarkov) or pd.isna(n) or n <= 0:
        return np.nan
    var = max(p_markov * (1.0 - p_markov), 0.0) / n
    var += max(p_nonmarkov * (1.0 - p_nonmarkov), 0.0) / n
    return 1.96 * math.sqrt(var)


def plot_accuracy_8lines(csv_path: Path, output_dir: Path, show: bool = False) -> None:
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "overall"].copy()
    df["rounds"] = pd.to_numeric(df["rounds"])
    df["samples"] = pd.to_numeric(df["samples"])

    rounds = sorted(df["rounds"].dropna().unique())
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    models += [m for m in df["model"].unique() if m not in models]

    panel_specs = [
        ("nonmarkov_acc", "(a) Non-Markov players", "accuracy"),
        ("markov_recall_strict", "(b) Markov players", "accuracy"),
        ("gap", "(c) Markov - Non-Markov", "gap"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.75))
    plot_rows = []

    for ax, (col, panel_title, panel_type) in zip(axes, panel_specs):
        for model in models:
            model_data = df[df["model"] == model].sort_values("rounds")
            color = MODEL_COLORS.get(model, "#333333")
            x, y, yerr = [], [], []
            for r in rounds:
                row = model_data[model_data["rounds"] == r]
                if row.empty:
                    continue
                n = float(row["samples"].iloc[0])
                if col == "gap":
                    markov_acc = float(row["markov_recall_strict"].iloc[0])
                    nonmarkov_acc = float(row["nonmarkov_acc"].iloc[0])
                    p = markov_acc - nonmarkov_acc
                    ci = diff_ci_95(markov_acc, nonmarkov_acc, n)
                else:
                    p = float(row[col].iloc[0])
                    ci = binomial_ci_95(p, n)
                x.append(r)
                y.append(p)
                yerr.append(ci)
                plot_rows.append(
                    {
                        "panel": panel_title,
                        "panel_type": panel_type,
                        "model": model,
                        "rounds": r,
                        "samples": n,
                        "value": p,
                        "ci95": ci,
                        "ci95_lower": p - ci if panel_type == "gap" else max(p - ci, 0.0),
                        "ci95_upper": p + ci if panel_type == "gap" else min(p + ci, 1.0),
                        "nonmarkov_acc": float(row["nonmarkov_acc"].iloc[0]),
                        "markov_recall_strict": float(row["markov_recall_strict"].iloc[0]),
                    }
                )

            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            yerr_arr = np.asarray(yerr, dtype=float)
            if panel_type == "gap":
                lower = y_arr - yerr_arr
                upper = y_arr + yerr_arr
            else:
                lower = np.clip(y_arr - yerr_arr, 0.0, 1.0)
                upper = np.clip(y_arr + yerr_arr, 0.0, 1.0)

            ax.fill_between(
                x_arr,
                lower,
                upper,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                x_arr,
                y_arr,
                color=color,
                linestyle="-",
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                alpha=0.95,
            )

        ax.set_title(panel_title, loc="left", pad=7, color="#111111")
        ax.set_xlabel("Context rounds", labelpad=6)
        ax.set_xticks(rounds)
        ax.set_facecolor("#f6f7f8")
        ax.grid(axis="y", color="#d5d9dd", alpha=0.95, linestyle="-", linewidth=0.72)
        ax.grid(axis="x", color="#e3e6e8", alpha=0.8, linestyle="-", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b8bdc2")
        ax.spines["bottom"].set_color("#b8bdc2")
        if panel_type == "gap":
            ax.axhline(0, color="#555555", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
            ax.set_ylim(-0.55, 0.55)
        else:
            ax.set_ylim(0.0, 1.02)

    axes[0].set_ylabel("Accuracy")
    axes[2].set_ylabel("Accuracy difference")

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS.get(model, "#333333"), lw=2.2, marker="o", markersize=4.2, label=MODEL_LABELS.get(model, model))
        for model in models
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.070, right=0.990, top=0.90, bottom=0.24, wspace=0.22)

    png_dir = output_dir / "others" / "png"
    pdf_dir = output_dir / "others" / "pdf"
    data_dir = output_dir / "others" / "data"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    stem = "accuracy_8lines_markov_nonmarkov_95ci"
    pd.DataFrame(plot_rows).to_csv(data_dir / f"{stem}_raw_data.csv", index=False)
    fig.savefig(png_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{stem}.pdf", dpi=400, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(png_dir / f"{stem}.png")
    print(pdf_dir / f"{stem}.pdf")
    print(data_dir / f"{stem}_raw_data.csv")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(root / "exp1(strategy)" / "metrics_export.csv"))
    parser.add_argument("--output-dir", default=str(root / "exp1(strategy)" / "plots"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_accuracy_8lines(Path(args.csv), Path(args.output_dir), show=args.show)


if __name__ == "__main__":
    main()
