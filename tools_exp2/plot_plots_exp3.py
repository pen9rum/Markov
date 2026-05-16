"""
Generate every figure in exp2(generation_blind)/plots_exp3.

Canonical output folder:
  exp2(generation_blind)/plots_exp3/

This combines the scripts that produced the historical plots_exp3 folder:
  - run_analysis.py
  - compute_cumulative_metrics_xyz.py
  - plot_cumulative_metrics_xyz.py
  - plot_identity_condition_cumulative_xyz.py

Usage:
  python tools_exp2/plot_plots_exp3.py
  python tools_exp2/plot_plots_exp3.py --model deepseek-reasoner
"""
import argparse
import os
import sys

import compute_cumulative_metrics_xyz
import plot_cumulative_metrics_xyz
import plot_identity_condition_cumulative_xyz
import run_analysis
from plot_output import expected_pdf_files, mirror_png_to_pdf, split_output_roots, verify_expected


EXP2_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
PLOT_ROOT = os.path.join(EXP2_ROOT, 'plots_exp3')
PNG_ROOT, PDF_ROOT = split_output_roots(PLOT_ROOT)

BASE_EXPECTED = {
    'acc_vs_rounds_overall.png',
    'confusion_identity.png',
    'confusion_markov.png',
    'confusion_matrix.png',
    'cumulative_overall.png',
    'cumulative_zoom_overlap_strict_overall.png',
    'generation_metrics.png',
    'identification_accuracy.png',
    'identity_condition.png',
    'identity_condition_cecum_by_model.png',
    'identity_condition_cumulative.png',
    'identity_condition_distribution.png',
    'identity_condition_distribution_by_model.png',
    'identity_condition_markov.png',
    'identity_condition_markov_by_model.png',
    'identity_condition_msecum_by_model.png',
    'identity_condition_overlapcum_by_model.png',
    'identity_condition_strictcum_by_model.png',
    'model_comparison_by_window.png',
}


def _existing_models():
    gen_root = os.path.join(EXP2_ROOT, 'generation')
    if not os.path.isdir(gen_root):
        return []
    return sorted(
        name for name in os.listdir(gen_root)
        if os.path.isdir(os.path.join(gen_root, name))
        and name in run_analysis.VALID_MODELS
    )


def _expected_files(model=None):
    models = [model] if model else _existing_models()
    expected = set(BASE_EXPECTED)
    expected.update(f'{m.replace("/", "_")}_slump.png' for m in models)
    expected.update(f'cumulative_{m.replace("/", "_")}.png' for m in models)
    return expected


def _run_analysis(model=None):
    run_analysis.PLOT_ROOT = PNG_ROOT
    argv = ['run_analysis.py']
    if model:
        argv.extend(['--model', model])

    old_argv = sys.argv
    try:
        sys.argv = argv
        run_analysis.main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description='Generate exp2 exp3-style plots')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter standard plots to one model name before cumulative plots are computed')
    args = parser.parse_args()

    with mirror_png_to_pdf(PNG_ROOT, PDF_ROOT):
        _run_analysis(model=args.model)
        compute_cumulative_metrics_xyz.main()

        plot_cumulative_metrics_xyz.PLOT_ROOT = PNG_ROOT
        plot_cumulative_metrics_xyz.main()

        plot_identity_condition_cumulative_xyz.PLOT_ROOT = PNG_ROOT
        plot_identity_condition_cumulative_xyz.main()

    expected = _expected_files(args.model)
    verify_expected(PNG_ROOT, expected)
    verify_expected(PDF_ROOT, expected_pdf_files(expected))


if __name__ == '__main__':
    main()
