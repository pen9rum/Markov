"""
Exp2 — Plot cumulative metrics per model and overall.
Same visual style as tools_exp3/plot_cumulative_metrics.py.

Input:  exp2(generation_blind)/analysis_results/xyz_cumulative_metrics.csv
Output: exp2(generation_blind)/plots/
          cumulative_{model}.png
          cumulative_overall.png
          cumulative_zoom_overlap_strict_overall.png

Usage:
  python tools_exp2/plot_cumulative_metrics_xyz.py
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

MARKOV_SET   = 'xyz'
COMBO_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
COMBO_NAMES  = {1: 'Type1: NonMarkov', 2: 'Type2: Markov_P1', 3: 'Type3: Markov_P2'}
MODEL_COLORS = {
    'deepseek-chat':     '#1f77b4',
    'deepseek-reasoner': '#ff7f0e',
    'gpt-5':             '#2ca02c',
    'gpt-5-mini':        '#d62728',
}


def to_float(x):
    try: return float(x)
    except Exception: return None


def avg_by_window(rows, metric):
    by_w = defaultdict(list)
    for r in rows:
        v = to_float(r.get(metric))
        if v is not None:
            by_w[int(r['window_idx'])].append(v)
    xs = sorted(by_w)
    ys = [sum(by_w[x]) / len(by_w[x]) for x in xs]
    return xs, ys


def setup_ax(ax, title, ylabel, ylim=None, baseline=None, windows=None):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Window Index (100 rounds per window)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim: ax.set_ylim(*ylim)
    if windows: ax.set_xticks(windows)
    if baseline is not None:
        ax.axhline(baseline, color='purple', linestyle='--', linewidth=1.2, label='Random (1/3)')
    handles, _ = ax.get_legend_handles_labels()
    if handles: ax.legend(fontsize=8)


def plot_per_model(rows, outdir):
    models  = sorted({r['model'] for r in rows})
    windows = sorted({int(r['window_idx']) for r in rows})

    for model in models:
        m_rows = [r for r in rows if r['model'] == model]
        combos = sorted({int(r['combo_type']) for r in m_rows})
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for c in combos:
            c_rows = [r for r in m_rows if int(r['combo_type']) == c]
            color  = COMBO_COLORS.get(c)
            label  = COMBO_NAMES.get(c, f'combo_{c}')
            for ax_i, metric, ylabel, ylim, bl in [
                (0, 'ce_cum',      'Cumulative CE',      None,      None),
                (1, 'mse_cum',     'Cumulative MSE',     None,      None),
                (2, 'overlap_cum', 'Cumulative Overlap', (0, 1.05), 1/3),
                (3, 'strict_cum',  'Cumulative Strict',  (0, 1.05), None),
            ]:
                xs, ys = avg_by_window(c_rows, metric)
                if xs:
                    axes[ax_i].plot(xs, ys, marker='o', linewidth=2, color=color, label=label)

        titles = [
            f'{model} — Cumulative CE ({MARKOV_SET.upper()})',
            f'{model} — Cumulative MSE ({MARKOV_SET.upper()})',
            f'{model} — Cumulative Overlap ({MARKOV_SET.upper()})',
            f'{model} — Cumulative Strict ({MARKOV_SET.upper()})',
        ]
        for ax, title, ylabel, ylim, bl in zip(axes, titles,
            ['CE', 'MSE', 'Overlap Rate', 'Strict Rate'],
            [None, None, (0, 1.05), (0, 1.05)],
            [None, None, 1/3, None]):
            setup_ax(ax, title, ylabel, ylim, bl, windows)

        plt.tight_layout()
        clean = model.replace('/', '_')
        path = os.path.join(outdir, f'cumulative_{clean}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Plot -> {path}')


def plot_overall(rows, outdir):
    models  = sorted({r['model'] for r in rows})
    windows = sorted({int(r['window_idx']) for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for model in models:
        m_rows = [r for r in rows if r['model'] == model]
        color  = MODEL_COLORS.get(model)
        for ax_i, metric in enumerate(['ce_cum', 'mse_cum', 'overlap_cum', 'strict_cum']):
            xs, ys = avg_by_window(m_rows, metric)
            if xs:
                axes[ax_i].plot(xs, ys, marker='o', linewidth=2, color=color, label=model)

    titles = [
        f'All Models — Cumulative CE ({MARKOV_SET.upper()})',
        f'All Models — Cumulative MSE ({MARKOV_SET.upper()})',
        f'All Models — Cumulative Overlap ({MARKOV_SET.upper()})',
        f'All Models — Cumulative Strict ({MARKOV_SET.upper()})',
    ]
    for ax, title, ylabel, ylim, bl in zip(axes, titles,
        ['CE', 'MSE', 'Overlap Rate', 'Strict Rate'],
        [None, None, (0, 1.05), (0, 1.05)],
        [None, None, 1/3, None]):
        setup_ax(ax, title, ylabel, ylim, bl, windows)

    plt.tight_layout()
    path = os.path.join(outdir, 'cumulative_overall.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _tight_ylim(vals, pad=0.01, lo=0.0, hi=1.05):
    vs = [v for v in vals if v is not None]
    if not vs:
        return (lo, hi)
    vmin, vmax = min(vs), max(vs)
    if abs(vmax - vmin) < 1e-12:
        return (max(lo, vmin - 0.02), min(hi, vmax + 0.02))
    return (max(lo, vmin - pad), min(hi, vmax + pad))


def plot_zoomed_overlap_strict(rows, outdir):
    models  = sorted({r['model'] for r in rows})
    windows = sorted({int(r['window_idx']) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    overlap_all, strict_all = [], []

    for model in models:
        m_rows = [r for r in rows if r['model'] == model]
        color  = MODEL_COLORS.get(model)
        xs_o, ys_o = avg_by_window(m_rows, 'overlap_cum')
        xs_s, ys_s = avg_by_window(m_rows, 'strict_cum')
        if xs_o:
            axes[0].plot(xs_o, ys_o, marker='o', linewidth=2, color=color, label=model)
            overlap_all.extend(ys_o)
        if xs_s:
            axes[1].plot(xs_s, ys_s, marker='o', linewidth=2, color=color, label=model)
            strict_all.extend(ys_s)

    overlap_ylim = (0.9, 1.0)
    strict_ylim  = _tight_ylim(strict_all, pad=0.01, lo=0.0, hi=1.05)

    setup_ax(axes[0], f'All Models — Cumulative Overlap (Zoom) ({MARKOV_SET.upper()})',
             'Overlap Rate', overlap_ylim, 1/3, windows)
    setup_ax(axes[1], f'All Models — Cumulative Strict (Zoom) ({MARKOV_SET.upper()})',
             'Strict Rate', strict_ylim, None, windows)

    plt.tight_layout()
    path = os.path.join(outdir, 'cumulative_zoom_overlap_strict_overall.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def main():
    in_path = os.path.join(RESULT_ROOT, 'xyz_cumulative_metrics.csv')
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run compute_cumulative_metrics_xyz.py first.")
        return

    with open(in_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    os.makedirs(PLOT_ROOT, exist_ok=True)
    plot_per_model(rows, PLOT_ROOT)
    plot_overall(rows, PLOT_ROOT)
    plot_zoomed_overlap_strict(rows, PLOT_ROOT)
    print(f'\nDone. Plots -> {PLOT_ROOT}')


if __name__ == '__main__':
    main()
