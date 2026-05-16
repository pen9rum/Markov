"""
Generate every figure in exp2(generation_blind)/plots.

Canonical output folder:
  exp2(generation_blind)/plots/

This entry point covers the current standard exp2 figures. It delegates the
metric refresh and plotting to run_analysis.py, then verifies that all expected
plot files were produced.

Usage:
  python tools_exp2/plot_plots.py
  python tools_exp2/plot_plots.py --model deepseek-reasoner
"""
import argparse
import os
import sys

import run_analysis
from plot_output import expected_pdf_files, mirror_png_to_pdf, split_output_roots, verify_expected


EXP2_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
PLOT_ROOT = os.path.join(EXP2_ROOT, 'plots')
PNG_ROOT, PDF_ROOT = split_output_roots(PLOT_ROOT)

BASE_EXPECTED = {
    'acc_vs_rounds_overall.png',
    'confusion_identity.png',
    'confusion_markov.png',
    'confusion_matrix.png',
    'generation_metrics.png',
    'identification_accuracy.png',
    'identity_condition.png',
    'identity_condition_distribution.png',
    'identity_condition_distribution_by_model.png',
    'identity_condition_markov.png',
    'identity_condition_markov_by_model.png',
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
    return BASE_EXPECTED | {f'{m.replace("/", "_")}_slump.png' for m in models}


def main():
    parser = argparse.ArgumentParser(description='Generate exp2 standard plots')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter to one model name (default: all models)')
    args = parser.parse_args()

    run_analysis.PLOT_ROOT = PNG_ROOT
    argv = ['run_analysis.py']
    if args.model:
        argv.extend(['--model', args.model])

    old_argv = sys.argv
    try:
        sys.argv = argv
        with mirror_png_to_pdf(PNG_ROOT, PDF_ROOT):
            run_analysis.main()
    finally:
        sys.argv = old_argv

    expected = _expected_files(args.model)
    verify_expected(PNG_ROOT, expected)
    verify_expected(PDF_ROOT, expected_pdf_files(expected))


if __name__ == '__main__':
    main()
