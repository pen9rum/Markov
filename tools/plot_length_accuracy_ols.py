import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, t


MODELS = ["deepseek-chat", "deepseek-reasoner", "gpt-5", "gpt-5-mini"]
ROUNDS = ["100", "200", "500", "1000"]
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
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _extract_section(text: str, start_label: str, end_label: str | None = None) -> str:
    start = text.find(start_label)
    if start < 0:
        return ""
    start += len(start_label)
    if end_label is None:
        return text[start:].strip()
    end = text.find(end_label, start)
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()


def estimate_input_length(report_text: str) -> int:
    """Estimate the actual prompt length from the saved report."""
    before_llm = report_text.split("LLM Analysis:", 1)[0]
    p1 = _extract_section(before_llm, "Player1 (", "\n\nPlayer2 (")
    p2 = _extract_section(before_llm, "Player2 (", "\n\nPlayer1 Actual Distribution:")
    game_info = _extract_section(before_llm, "Game Results", "Actual Game Data")
    # Use the report prompt-bearing content as a stable character-length proxy.
    return len(game_info) + len(p1) + len(p2)


def estimate_output_length(report_text: str) -> int:
    return len(_extract_section(report_text, "LLM Analysis:", None))


def collect_rows(root: Path) -> pd.DataFrame:
    parsed_base = root / "exp1(strategy)" / "parsed_output"
    batch_base = root / "exp1(strategy)" / "batch_results"
    rows = []

    for model in MODELS:
        for rounds in ROUNDS:
            detail_path = parsed_base / model / rounds / "evaluation_detail.json"
            if not detail_path.exists():
                continue
            data = json.loads(detail_path.read_text(encoding="utf-8"))
            for row in data.get("rows", []):
                parsed_file = Path(row["file"])
                rel_parts = parsed_file.parts
                try:
                    idx = rel_parts.index("parsed_output")
                    rel_after = Path(*rel_parts[idx + 1 :])
                except ValueError:
                    rel_after = Path(model) / rounds / Path(row["file"]).parent.name / Path(row["file"]).name

                txt_rel = rel_after.with_suffix(".txt")
                txt_path = batch_base / txt_rel
                if not txt_path.exists():
                    # Fall back by matching player pair if timestamps differ.
                    match = re.match(r"analysis_([A-Z])_vs_([A-Z])_", txt_rel.name)
                    if not match:
                        continue
                    pattern = f"analysis_{match.group(1)}_vs_{match.group(2)}_*.txt"
                    candidates = list((batch_base / model / rounds / txt_rel.parent.name).glob(pattern))
                    if not candidates:
                        continue
                    txt_path = candidates[0]

                report = txt_path.read_text(encoding="utf-8", errors="replace")
                rows.append(
                    {
                        "model": model,
                        "rounds": int(rounds),
                        "combo_type": txt_rel.parent.name,
                        "acc": float(row.get("ACC", np.nan)),
                        "input_length": estimate_input_length(report),
                        "output_length": estimate_output_length(report),
                    }
                )

    return pd.DataFrame(rows).dropna(subset=["acc", "input_length", "output_length"])


def grouped_accuracy_ci(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = (
        df.groupby(group_col)["acc"]
        .agg(mean_acc="mean", n="count")
        .reset_index()
    )
    agg["ci95"] = 1.96 * np.sqrt(agg["mean_acc"] * (1.0 - agg["mean_acc"]) / agg["n"])
    return agg


def ols_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    fit = linregress(x, y)
    slope = fit.slope
    intercept = fit.intercept
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "intercept": intercept,
        "slope": slope,
        "se": fit.stderr,
        "intercept_se": fit.intercept_stderr,
        "p_value": fit.pvalue,
        "r2": r2,
        "n": len(x),
        "x": x,
        "y": y,
    }


def style_axis(ax) -> None:
    ax.grid(True, axis="y", linestyle="-", linewidth=0.45, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0, 1, 6))


def save_figure(fig, output_dir: Path, out_stem: str) -> None:
    png_dir = output_dir / "others" / "png"
    pdf_dir = output_dir / "others" / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{out_stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{out_stem}.pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)


def ols_mean_ci(stats: dict[str, float | np.ndarray], x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(stats["x"], dtype=float)
    y = np.asarray(stats["y"], dtype=float)
    n = len(x)
    y_hat = float(stats["intercept"]) + float(stats["slope"]) * x_grid
    if n <= 2:
        return y_hat, np.zeros_like(y_hat)
    resid = y - (float(stats["intercept"]) + float(stats["slope"]) * x)
    s_err = math.sqrt(np.sum(resid**2) / (n - 2))
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    tcrit = t.ppf(0.975, n - 2)
    ci = tcrit * s_err * np.sqrt((1.0 / n) + ((x_grid - x_mean) ** 2 / sxx))
    return y_hat, ci


def add_ols_layer(
    ax,
    stats: dict[str, float | np.ndarray],
    x_min: float,
    x_max: float,
    color: str,
    x_scale: float = 1.0,
) -> None:
    x_grid = np.linspace(x_min, x_max, 240)
    y_hat, ci = ols_mean_ci(stats, x_grid)
    x_plot = x_grid / x_scale
    ax.plot(x_plot, y_hat, color=color, linewidth=1.7)
    ax.fill_between(
        x_plot,
        np.clip(y_hat - ci, 0, 1),
        np.clip(y_hat + ci, 0, 1),
        color=color,
        alpha=0.16,
        linewidth=0,
    )


def annotate_stats(ax, stats: dict[str, float | np.ndarray]) -> None:
    p = float(stats["p_value"])
    p_text = "<.001" if p < 0.001 else f"={p:.3f}"
    ax.text(
        0.04,
        0.08,
        f"β={float(stats['slope']):.2e}, p{p_text}\nR²={float(stats['r2']):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )


def plot_ols_facets(df: pd.DataFrame, x_col: str, out_stem: str, output_dir: Path) -> list[dict[str, float | str]]:
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.95), sharey=True)
    summary = []
    is_input = x_col == "input_length"

    for ax, model in zip(axes.ravel(), MODELS):
        sub = df[df["model"] == model].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        stats = ols_stats(sub[x_col].to_numpy(), sub["acc"].to_numpy())
        summary.append(
            {
                "figure": x_col,
                "model": model,
                "intercept": stats["intercept"],
                "slope": stats["slope"],
                "se": stats["se"],
                "intercept_se": stats["intercept_se"],
                "p_value": stats["p_value"],
                "r2": stats["r2"],
                "n": stats["n"],
            }
        )

        if is_input:
            agg = (
                sub.groupby("rounds")
                .agg(x_mean=(x_col, "mean"))
                .join(grouped_accuracy_ci(sub, "rounds").set_index("rounds"))
                .reset_index(drop=True)
                .sort_values("x_mean")
            )
            x_scale = 1000.0
        else:
            sub["length_bin"] = pd.qcut(sub[x_col], q=4, labels=False, duplicates="drop")
            agg = (
                sub.groupby("length_bin")
                .agg(x_mean=(x_col, "mean"))
                .join(grouped_accuracy_ci(sub, "length_bin").set_index("length_bin"))
                .reset_index(drop=True)
                .sort_values("x_mean")
            )
            x_scale = 1.0

        x = agg["x_mean"].to_numpy(dtype=float) / x_scale
        y = agg["mean_acc"].to_numpy(dtype=float)
        ci = agg["ci95"].to_numpy(dtype=float)
        color = MODEL_COLORS[model]

        ax.errorbar(
            x,
            y,
            yerr=ci,
            fmt="o",
            markersize=4.3,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.15,
            ecolor=color,
            elinewidth=0.85,
            capsize=2.0,
            alpha=0.95,
            zorder=3,
        )
        add_ols_layer(ax, stats, sub[x_col].min(), sub[x_col].max(), color=color, x_scale=x_scale)
        style_axis(ax)
        ax.set_title(MODEL_LABELS[model], color="#111111", pad=4)
        annotate_stats(ax, stats)

    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    for ax in axes[-1, :]:
        if is_input:
            ax.set_xlabel("Input length (thousand characters)")
        else:
            ax.set_xlabel("Output length (characters)")

    title = "OLS: input length and accuracy" if is_input else "OLS: output length and accuracy"
    fig.suptitle(title, y=0.985, fontsize=10.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.11, wspace=0.16, hspace=0.34)
    save_figure(fig, output_dir, out_stem)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "exp1(strategy)" / "plots"
    df = collect_rows(root)
    data_dir = output_dir / "others" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "length_accuracy_ols_data.csv", index=False)

    summary = []
    summary.extend(plot_ols_facets(df, "input_length", "ols_input_length_vs_accuracy", output_dir))
    summary.extend(plot_ols_facets(df, "output_length", "ols_output_length_vs_accuracy", output_dir))
    pd.DataFrame(summary).to_csv(data_dir / "length_accuracy_ols_by_model_summary.csv", index=False)

    print(f"rows: {len(df)}")
    print(output_dir / "others" / "png" / "ols_input_length_vs_accuracy.png")
    print(output_dir / "others" / "png" / "ols_output_length_vs_accuracy.png")
    print(output_dir / "others" / "pdf" / "ols_input_length_vs_accuracy.pdf")
    print(output_dir / "others" / "pdf" / "ols_output_length_vs_accuracy.pdf")


if __name__ == "__main__":
    main()
