"""
Generate every figure in exp3(complex_markov)/plots.

Canonical output folders:
  exp3(complex_markov)/plots/{qrs,tuv,xyz_opp}/png/
  exp3(complex_markov)/plots/{qrs,tuv,xyz_opp}/pdf/

Usage:
  python tools_exp3/plot_plots.py
  python tools_exp3/plot_plots.py --markov-set qrs
  python tools_exp3/plot_plots.py --markov-set qrs --model deepseek-reasoner
"""
import argparse
import os
import shutil
import sys

import compute_cumulative_metrics
import plot_cumulative_metrics
import plot_identity_condition_cumulative
import run_analysis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools_exp2'))
from plot_output import expected_pdf_files, mirror_png_to_pdf, split_output_roots, verify_expected


EXP3_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
PLOT_ROOT = os.path.join(EXP3_ROOT, 'plots')
MARKOV_SETS = ['qrs', 'tuv', 'xyz_opp']

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


def _models_for_set(markov_set, model=None):
    if model:
        return [model]
    gen_root = os.path.join(EXP3_ROOT, 'generation', markov_set)
    if not os.path.isdir(gen_root):
        return []
    return sorted(
        name for name in os.listdir(gen_root)
        if os.path.isdir(os.path.join(gen_root, name))
    )


def _expected_files(markov_set, model=None):
    models = _models_for_set(markov_set, model=model)
    expected = set(BASE_EXPECTED)
    expected.update(f'{m.replace("/", "_")}_slump.png' for m in models)
    expected.update(f'cumulative_{m.replace("/", "_")}.png' for m in models)
    return expected


def _call_main(module, argv):
    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv


def _move_root_pngs(set_root, png_root):
    os.makedirs(png_root, exist_ok=True)
    for name in os.listdir(set_root):
        src = os.path.join(set_root, name)
        if not os.path.isfile(src) or not name.lower().endswith('.png'):
            continue
        dst = os.path.join(png_root, name)
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)


def _run_one(markov_set, model=None):
    set_root = os.path.join(PLOT_ROOT, markov_set)
    png_root, pdf_root = split_output_roots(set_root)
    os.makedirs(set_root, exist_ok=True)

    with mirror_png_to_pdf(set_root, pdf_root):
        argv = ['run_analysis.py', '--markov-set', markov_set]
        if model:
            argv.extend(['--model', model])
        _call_main(run_analysis, argv)

        _call_main(compute_cumulative_metrics, [
            'compute_cumulative_metrics.py',
            '--markov-set',
            markov_set,
        ])
        _call_main(plot_cumulative_metrics, [
            'plot_cumulative_metrics.py',
            '--markov-set',
            markov_set,
        ])
        _call_main(plot_identity_condition_cumulative, [
            'plot_identity_condition_cumulative.py',
            '--markov-set',
            markov_set,
        ])

    _move_root_pngs(set_root, png_root)

    expected = _expected_files(markov_set, model=model)
    verify_expected(png_root, expected)
    verify_expected(pdf_root, expected_pdf_files(expected))


def main():
    parser = argparse.ArgumentParser(description='Generate exp3 plots with PNG/PDF outputs')
    parser.add_argument('--markov-set', choices=MARKOV_SETS, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    markov_sets = [args.markov_set] if args.markov_set else MARKOV_SETS
    for markov_set in markov_sets:
        print(f'\nGenerating exp3 plots for {markov_set} ...')
        _run_one(markov_set, model=args.model)

    print(f'\nDone. Exp3 plots -> {PLOT_ROOT}')


if __name__ == '__main__':
    main()
