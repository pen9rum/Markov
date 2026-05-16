"""
Generate every figure in exp2(generation_blind)/plots_exp1.

Canonical output folder:
  exp2(generation_blind)/plots_exp1/

Input:
  exp2(generation_blind)/analysis_results/xyz_metrics.csv

Run tools_exp2/plot_plots.py first if xyz_metrics.csv needs to be refreshed.

Usage:
  python tools_exp2/plot_plots_exp1.py
"""
import os

import run_exp1style_analysis
from plot_output import expected_pdf_files, mirror_png_to_pdf, split_output_roots, verify_expected


EXP2_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
PLOT_ROOT = os.path.join(EXP2_ROOT, 'plots_exp1')
PNG_ROOT, PDF_ROOT = split_output_roots(PLOT_ROOT)

METRICS = tuple(run_exp1style_analysis.METRIC_INFO)


def _expected_files():
    expected = {f'{metric}_vs_rounds_overall.png' for metric in METRICS}
    for folder in ('confusion_markov', 'confusion_identity'):
        expected.add(os.path.join(folder, 'confusion_overall_all.png'))
        for rounds in range(100, 1001, 100):
            expected.add(os.path.join(folder, f'confusion_overall_rounds{rounds}.png'))
    return expected


def main():
    run_exp1style_analysis.PLOT_ROOT = PNG_ROOT
    with mirror_png_to_pdf(PNG_ROOT, PDF_ROOT):
        run_exp1style_analysis.main()

    expected = _expected_files()
    verify_expected(PNG_ROOT, expected)
    verify_expected(PDF_ROOT, expected_pdf_files(expected))


if __name__ == '__main__':
    main()
