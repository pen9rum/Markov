"""
Exp2 — Plot cumulative metrics conditioned on identity correctness.
Same visual style as exp3/plot_identity_condition_cumulative.py.

Input:  exp2(generation_blind)/analysis_results/cumulative_metrics.csv
Output: exp2(generation_blind)/plots/
          identity_condition_cumulative.png
          identity_condition_{metric}_by_model.png

Usage:
  python tools_exp2/plot_identity_condition_cumulative_v2.py
"""
import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP2_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
RESULT_ROOT = os.path.join(EXP2_ROOT, 'analysis_results')
PLOT_ROOT   = os.path.join(EXP2_ROOT, 'plots')

MODEL_COLORS = {
    'deepseek-chat': '#1f77b4', 'deepseek-reasoner': '#ff7f0e',
    'gpt-5': '#2ca02c',         'gpt-5-mini': '#d62728',
}
MARKER = 'o'

# exp2 column name mappings (exp3 uses 'overlap_cum' and 'both_correct')
OVERLAP_COL  = 'exact_cum'
ID_MATCH_COL = 'id_match_both'


def to_float(x):
    try: return float(x)
    except Exception: return None


def identity_correct(row):
    try: return int(row.get(ID_MATCH_COL, ''))
    except Exception: return None


def group_avg(rows, metric, condition=None, model=None):
    by_w = defaultdict(list)
    for r in rows:
        if model and r.get('model') != model:
            continue
        if condition is not None:
            c = condition(r)
            if c is None or c != 1:
                continue
        v = to_float(r.get(metric))
        if v is None: continue
        by_w[int(r['window_idx'])].append(v)
    xs = sorted(by_w)
    ys = [sum(by_w[x]) / len(by_w[x]) for x in xs]
    return xs, ys


def group_avg_wrong(rows, metric, model=None):
    by_w = defaultdict(list)
    for r in rows:
        if model and r.get('model') != model:
            continue
        c = identity_correct(r)
        if c is None or c != 0: continue
        v = to_float(r.get(metric))
        if v is None: continue
        by_w[int(r['window_idx'])].append(v)
    xs = sorted(by_w)
    ys = [sum(by_w[x]) / len(by_w[x]) for x in xs]
    return xs, ys


def plot_combined(rows, outdir):
    metrics = [
        (OVERLAP_COL,  'Cumulative Overlap Rate', (0, 1.05)),
        ('strict_cum', 'Cumulative Strict Rate',  (0, 1.05)),
        ('ce_cum',     'Cumulative CE',            None),
        ('mse_cum',    'Cumulative MSE',           None),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        cx, cy = group_avg(rows, metric, condition=lambda r: identity_correct(r))
        wx, wy = group_avg_wrong(rows, metric)
        if cx: ax.plot(cx, cy, marker=MARKER, linewidth=2, color='#2E86DE', label='Identity Correct')
        if wx: ax.plot(wx, wy, marker=MARKER, linewidth=2, color='#E74C3C', label='Identity Wrong')
        ax.set_title(f'{ylabel} — XYZ')
        ax.set_xlabel('Window Index')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(outdir, 'identity_condition_cumulative.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_by_model(rows, outdir):
    models = sorted({r['model'] for r in rows})
    metrics = [
        (OVERLAP_COL,  'Cumulative Overlap', (0, 1.05)),
        ('strict_cum', 'Cumulative Strict',  (0, 1.05)),
        ('ce_cum',     'Cumulative CE',       None),
        ('mse_cum',    'Cumulative MSE',      None),
    ]

    for metric, ylabel, ylim in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, model in enumerate(models[:4]):
            ax    = axes[idx]
            color = MODEL_COLORS.get(model, '#2E86DE')
            cx, cy = group_avg(rows, metric, condition=lambda r: identity_correct(r), model=model)
            wx, wy = group_avg_wrong(rows, metric, model=model)
            if cx: ax.plot(cx, cy, marker=MARKER, linewidth=2, color=color,    label='Identity Correct')
            if wx: ax.plot(wx, wy, marker=MARKER, linewidth=2, color='#E74C3C', label='Identity Wrong')
            ax.set_title(f'{model} — {ylabel}')
            ax.set_xlabel('Window Index')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if ylim: ax.set_ylim(*ylim)
            ax.legend(fontsize=8)

        for ax in axes[len(models):]:
            ax.set_visible(False)

        plt.suptitle(f'{ylabel} by Identity — XYZ', fontsize=12, y=1.01)
        plt.tight_layout()
        safe = metric.replace('_', '')
        path = os.path.join(outdir, f'identity_condition_{safe}_by_model.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Plot -> {path}')


def main():
    in_path = os.path.join(RESULT_ROOT, 'cumulative_metrics.csv')
    if not os.path.exists(in_path):
        print(f'ERROR: {in_path} not found. Run compute_cumulative_metrics.py first.')
        return

    with open(in_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    os.makedirs(PLOT_ROOT, exist_ok=True)
    plot_combined(rows, outdir=PLOT_ROOT)
    plot_by_model(rows, outdir=PLOT_ROOT)
    print(f'\nDone. Plots -> {PLOT_ROOT}')


if __name__ == '__main__':
    main()
