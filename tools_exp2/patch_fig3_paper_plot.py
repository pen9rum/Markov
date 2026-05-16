"""Patch Exp2 paper Figure 3 using the original cached paper-plot pipeline.

The source file for tools_exp2/plot_paper_plots.py is missing from the working
tree, but its .pyc cache remains. This script loads that cached module and
replaces only Figure 3 so the original pipeline logic is otherwise preserved.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools_exp2"
PYCACHE = TOOLS / "__pycache__"


def _load_pyc(module_name: str, pyc_name: str):
    path = PYCACHE / pyc_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_original_paper_module():
    sys.path.insert(0, str(TOOLS))
    # Dependencies imported by the cached plot_paper_plots.pyc.
    _load_pyc("compute_cumulative_metrics_xyz", "compute_cumulative_metrics_xyz.cpython-313.pyc")
    _load_pyc("run_analysis", "run_analysis.cpython-313.pyc")
    _load_pyc("plot_output", "plot_output.cpython-313.pyc")
    return _load_pyc("plot_paper_plots_cached", "plot_paper_plots.cpython-313.pyc")


def patched_plot_main_strict_cumulative_dynamics(orig, cum_rows, outdir):
    """Main Figure 3: cumulative strict rule matching with context-length x axis."""
    models = orig._models_present(cum_rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    for model in models:
        by_w = defaultdict(list)
        for row in cum_rows:
            if row.get("model") != model:
                continue
            value = orig._to_float(row.get("strict_cum"))
            window_idx = orig._to_int(row.get("window_idx"))
            if value is None or window_idx is None:
                continue
            by_w[window_idx].append(value)

        xs_arr, ys_arr, _ci_arr = orig._mean_ci_arrays(by_w)
        context_lengths = np.asarray(xs_arr, dtype=float) * 100.0
        ax.plot(
            context_lengths,
            ys_arr,
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=orig.MODEL_COLORS[model],
            label=orig.MODEL_LABELS[model],
        )

    ax.set_title("Cumulative strict rule match")
    ax.set_ylabel("Cumulative strict rate")
    ax.set_xlabel("Context length (rounds)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.8)

    path = Path(outdir) / "fig3_cumulative_strict_rule_match.png"
    fig.savefig(path, dpi=orig.EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {path}")


def main() -> None:
    orig = load_original_paper_module()
    exp2_root = ROOT / "exp2(generation_blind)"
    orig.EXP2_ROOT = str(exp2_root)
    orig.RESULT_ROOT = str(exp2_root / "analysis_results")
    orig.PAPER_ROOT = str(exp2_root / "paper_plots")
    orig.MAIN_ROOT = str(exp2_root / "paper_plots" / "main")
    orig.APPENDIX_ROOT = str(exp2_root / "paper_plots" / "appendix")
    orig.MAIN_PNG = str(exp2_root / "paper_plots" / "main" / "png")
    orig.MAIN_PDF = str(exp2_root / "paper_plots" / "main" / "pdf")
    orig.APPENDIX_PNG = str(exp2_root / "paper_plots" / "appendix" / "png")
    orig.APPENDIX_PDF = str(exp2_root / "paper_plots" / "appendix" / "pdf")
    orig.METRICS_CSV = str(exp2_root / "analysis_results" / "xyz_metrics.csv")
    orig.CUMULATIVE_CSV = str(exp2_root / "analysis_results" / "xyz_cumulative_metrics.csv")

    cum_rows = orig._read_csv(orig.CUMULATIVE_CSV)

    # Generate only Figure 3 into the original main PNG folder, then mirror to PDF
    # with the original cached helper.
    patched_plot_main_strict_cumulative_dynamics(orig, cum_rows, orig.MAIN_PNG)
    orig.mirror_png_to_pdf(orig.MAIN_PNG, orig.MAIN_PDF, ["fig3_cumulative_strict_rule_match.png"])

    # Keep the current main/appendix organization from this working session:
    # fig2 and fig2-2 live in appendix, while the delta Figure 2 lives in main.
    print(Path(orig.MAIN_PNG) / "fig3_cumulative_strict_rule_match.png")
    print(Path(orig.MAIN_PDF) / "fig3_cumulative_strict_rule_match.pdf")


if __name__ == "__main__":
    main()
