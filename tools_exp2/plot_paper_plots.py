"""
Generate publication-focused exp2 figures.

This is intentionally separate from plots/, plots_exp1/, and plots_exp3/.
Those folders remain complete analysis exports; this script creates a smaller
paper-ready set with cleaner typography and figure organization.

Output:
  exp2(generation_blind)/paper_plots_exp2/main/png/
  exp2(generation_blind)/paper_plots_exp2/main/pdf/
  exp2(generation_blind)/paper_plots_exp2/appendix/png/
  exp2(generation_blind)/paper_plots_exp2/appendix/pdf/

Usage:
  python tools_exp2/plot_paper_plots.py
"""
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import compute_cumulative_metrics_xyz
import run_analysis
from plot_output import EXPORT_DPI, expected_pdf_files, mirror_png_to_pdf, split_output_roots, verify_expected


EXP2_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
RESULT_ROOT = os.path.join(EXP2_ROOT, 'analysis_results')
PAPER_ROOT = os.path.join(EXP2_ROOT, 'paper_plots_exp2')
MAIN_ROOT = os.path.join(PAPER_ROOT, 'main')
APPENDIX_ROOT = os.path.join(PAPER_ROOT, 'appendix')
MAIN_PNG, MAIN_PDF = split_output_roots(MAIN_ROOT)
APPENDIX_PNG, APPENDIX_PDF = split_output_roots(APPENDIX_ROOT)

METRICS_CSV = os.path.join(RESULT_ROOT, 'xyz_metrics.csv')
CUMULATIVE_CSV = os.path.join(RESULT_ROOT, 'xyz_cumulative_metrics.csv')

MARKOV_IDS = {'X', 'Y', 'Z'}
VALID_IDS = list('ABCDEFGHIJKLMNOPXYZ')
MODEL_ORDER = ['deepseek-chat', 'deepseek-reasoner', 'gpt-5', 'gpt-5-mini']
MODEL_LABELS = {
    'deepseek-chat': 'deepseek-chat',
    'deepseek-reasoner': 'deepseek-reasoner',
    'gpt-5': 'GPT-5',
    'gpt-5-mini': 'GPT-5-mini',
}
MODEL_TICK_LABELS = {
    'deepseek-chat': 'deepseek-\nchat',
    'deepseek-reasoner': 'deepseek-\nreasoner',
    'gpt-5': 'GPT-5',
    'gpt-5-mini': 'GPT-5-mini',
}
MODEL_COLORS = {
    'deepseek-chat': '#4477AA',
    'deepseek-reasoner': '#EE6677',
    'gpt-5': '#228833',
    'gpt-5-mini': '#CCBB44',
}
COND_COLORS = {
    'correct': '#4477AA',
    'wrong': '#EE6677',
    'overall': '#777777',
}


def _setup_style():
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _to_float(x):
    try:
        if x in (None, ''):
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x):
    try:
        if x in (None, ''):
            return None
        return int(float(x))
    except Exception:
        return None


def _mean_ci(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0, 0.0, 0
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0, len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    ci = 1.96 * math.sqrt(var / len(vals))
    return mean, ci, len(vals)


def _mean_ci_arrays(by_w):
    xs = sorted(by_w)
    ys, cis = [], []
    for w in xs:
        mean, ci, _ = _mean_ci(by_w[w])
        ys.append(mean)
        cis.append(ci)
    return np.array(xs), np.array(ys), np.array(cis)


def _read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run tools_exp2/plot_plots.py first.')
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _ensure_cumulative_csv():
    compute_cumulative_metrics_xyz.main()


def _models_present(rows):
    present = {r.get('model') for r in rows}
    return [m for m in MODEL_ORDER if m in present]


def plot_main_rule_following(rows, outdir):
    """Main Figure 1: Markov rule-following by Markov-side identity correctness."""
    markov_rows = [r for r in rows if _to_float(r.get('overlap_rate')) is not None]
    models = _models_present(markov_rows)

    metric_specs = [
        ('overlap_rate', 'Rule overlap', 'Overlap rate', 1 / 3),
        ('strict_exact', 'Strict rule match', 'Strict rate', None),
    ]
    cond_specs = [
        (1, 'Correct', COND_COLORS['correct']),
        (0, 'Wrong', COND_COLORS['wrong']),
        (None, 'Overall', COND_COLORS['overall']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    x = np.arange(len(models))
    width = 0.23
    offsets = (np.arange(len(cond_specs)) - (len(cond_specs) - 1) / 2) * width

    for ax, (col, title, ylabel, baseline) in zip(axes, metric_specs):
        for offset, (cond_value, label, color) in zip(offsets, cond_specs):
            means, cis = [], []
            for model in models:
                vals = []
                for r in markov_rows:
                    if r.get('model') != model:
                        continue
                    if cond_value is not None and _to_int(r.get('markov_id_match')) != cond_value:
                        continue
                    vals.append(_to_float(r.get(col)))
                mean, ci, _ = _mean_ci(vals)
                means.append(mean)
                cis.append(ci)
            ax.bar(x + offset, means, width, yerr=cis, capsize=2.2,
                   color=color, edgecolor='black', linewidth=0.45, label=label)
        if baseline is not None:
            ax.axhline(baseline, color='0.35', linestyle='--', linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_TICK_LABELS[m] for m in models])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelleft=True)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='0.35', linestyle='--', linewidth=1.0))
    labels.append('random baseline')
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, fontweight='bold',
                ha='left', va='top')
    fig.text(0.985, 0.02, 'error bars: 95% CI', ha='right', va='bottom', fontsize=7, color='0.35')
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.8)
    path = os.path.join(outdir, 'fig1_rule_following_summary.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _window_condition_series(rows, metric, condition_value, condition_col='both_correct'):
    by_w = defaultdict(list)
    for r in rows:
        cond = _to_int(r.get(condition_col))
        value = _to_float(r.get(metric))
        if value is None:
            continue
        if condition_value is None and cond is not None:
            by_w[_to_int(r.get('window_idx'))].append(value)
        elif condition_value is not None and cond == condition_value:
            by_w[_to_int(r.get('window_idx'))].append(value)
    xs = sorted(w for w in by_w if w is not None)
    ys, cis = [], []
    for w in xs:
        mean, ci, _ = _mean_ci(by_w[w])
        ys.append(mean)
        cis.append(ci)
    return np.array(xs), np.array(ys), np.array(cis)


def _condition_values_by_window(rows, metric, condition_value, condition_col):
    by_w = defaultdict(list)
    for r in rows:
        cond = _to_int(r.get(condition_col))
        value = _to_float(r.get(metric))
        if value is None:
            continue
        if condition_value is None and cond is not None:
            by_w[_to_int(r.get('window_idx'))].append(value)
        elif condition_value is not None and cond == condition_value:
            by_w[_to_int(r.get('window_idx'))].append(value)
    return by_w


def plot_main_identity_condition(cum_rows, outdir):
    """Main Figure 2: generation quality conditioned on identity correctness."""
    metric_specs = [
        ('overlap_cum', 'Rule overlap', 'Cumulative overlap', (0, 1.05)),
        ('strict_cum', 'Strict rule match', 'Cumulative strict rate', (0, 1.05)),
        ('ce_cum', 'Distribution CE', 'Cumulative CE', None),
        ('mse_cum', 'Distribution MSE', 'Cumulative MSE', None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    axes = axes.flatten()

    for idx, (ax, (metric, title, ylabel, ylim)) in enumerate(zip(axes, metric_specs)):
        for cond_value, label, color in [
            (1, 'Identity correct', COND_COLORS['correct']),
            (0, 'Identity wrong', COND_COLORS['wrong']),
        ]:
            xs, ys, cis = _window_condition_series(cum_rows, metric, cond_value)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, marker='o', linewidth=1.8, markersize=4, color=color, label=label)
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.13, linewidth=0)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if idx >= 2:
            ax.set_xlabel('Window index')
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(range(1, 11))
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for label, ax in zip(['(a)', '(b)', '(c)', '(d)'], axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight='bold', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.8)
    path = os.path.join(outdir, 'figA3_identity_condition_effect.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_main_identity_condition_nonmarkov(cum_rows, outdir):
    """Main Figure 2-2: distribution metrics by Markov/non-Markov identity correctness."""
    metric_specs = [
        ('ce_cum', 'Distribution CE', 'Cumulative CE'),
        ('mse_cum', 'Distribution MSE', 'Cumulative MSE'),
    ]
    line_specs = [
        ('markov_correct', None, 'Markov', 'Overall', 'Markov overall', '#444444', '-'),
        ('markov_correct', 1, 'Markov', 'Correct', 'Markov correct', '#4477AA', '-'),
        ('markov_correct', 0, 'Markov', 'Incorrect', 'Markov incorrect', '#4477AA', '--'),
        ('nonmarkov_correct', None, 'Non-Markov', 'Overall', 'Non-Markov overall', '#777777', ':'),
        ('nonmarkov_correct', 1, 'Non-Markov', 'Correct', 'Non-Markov correct', '#EE6677', '-'),
        ('nonmarkov_correct', 0, 'Non-Markov', 'Incorrect', 'Non-Markov incorrect', '#EE6677', '--'),
    ]
    raw_windows = {1, 2, 5, 10}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes = axes.flatten()
    raw_rows = []

    for ax, (metric, title, ylabel) in zip(axes, metric_specs):
        for condition_col, cond_value, side, condition_label, label, color, linestyle in line_specs:
            by_w = _condition_values_by_window(cum_rows, metric, cond_value, condition_col)
            xs, ys, cis = _mean_ci_arrays(by_w)
            if len(xs) == 0:
                continue
            for w, mean, ci in zip(xs, ys, cis):
                if int(w) not in raw_windows:
                    continue
                n = len(by_w[int(w)])
                raw_rows.append({
                    'figure': 'figA4_identity_condition_probability_side',
                    'metric': metric,
                    'metric_label': title,
                    'side': side,
                    'condition': condition_label,
                    'window_idx': int(w),
                    'window_end': int(w) * 100,
                    'mean': mean,
                    'ci95': ci,
                    'lower95': mean - ci,
                    'upper95': mean + ci,
                    'n': n,
                })
            ax.plot(xs, ys, marker='o', linewidth=1.5, markersize=3,
                    color=color, linestyle=linestyle, label=label)
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.055, linewidth=0)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Window index')
        ax.set_xticks(range(1, 11))
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight='bold', va='top')
    fig.text(0.985, 0.02, 'shaded bands: 95% CI; lower is better',
             ha='right', va='bottom', fontsize=7, color='0.35')
    fig.tight_layout(rect=[0, 0, 1, 0.86], pad=0.8)
    path = os.path.join(outdir, 'figA4_identity_condition_probability_side.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')
    raw_path = os.path.join(outdir, 'figA4_identity_condition_probability_side_raw.csv')
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'figure', 'metric', 'metric_label', 'side', 'condition',
            'window_idx', 'window_end', 'mean', 'ci95', 'lower95', 'upper95', 'n',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f'  Data -> {raw_path}')


def _distribution_delta_rows(cum_rows):
    raw_windows = {1, 2, 5, 10}
    metric_specs = [
        ('ce_cum', 'Distribution CE'),
        ('mse_cum', 'Distribution MSE'),
    ]
    condition_specs = [
        ('Overall', None, '#111111', 'o'),
        ('Correct', 1, '#D62728', 's'),
        ('Incorrect', 0, '#1F77B4', '^'),
    ]

    raw_rows = []
    for metric, metric_label in metric_specs:
        for condition_label, condition_value, _, _ in condition_specs:
            markov_by_w = _condition_values_by_window(cum_rows, metric, condition_value, 'markov_correct')
            nonmarkov_by_w = _condition_values_by_window(cum_rows, metric, condition_value, 'nonmarkov_correct')
            for w in sorted(raw_windows):
                markov_vals = markov_by_w.get(w, [])
                nonmarkov_vals = nonmarkov_by_w.get(w, [])
                markov_mean, _, markov_n = _mean_ci(markov_vals)
                nonmarkov_mean, _, nonmarkov_n = _mean_ci(nonmarkov_vals)
                if markov_n == 0 or nonmarkov_n == 0 or nonmarkov_mean == 0:
                    continue
                delta_pct = (markov_mean - nonmarkov_mean) / nonmarkov_mean * 100.0
                raw_rows.append({
                    'metric': metric,
                    'metric_label': metric_label,
                    'condition': condition_label,
                    'window_end': w * 100,
                    'markov_mean': markov_mean,
                    'nonmarkov_mean': nonmarkov_mean,
                    'delta_pct': delta_pct,
                    'markov_n': markov_n,
                    'nonmarkov_n': nonmarkov_n,
                })
    return raw_rows, condition_specs


def _plot_distribution_delta_panel(ax, raw_rows, condition_specs, metric, title, show_legend=False):
    metric_rows = [r for r in raw_rows if r['metric'] == metric]
    for condition_label, _, color, marker in condition_specs:
        rows_for_condition = [r for r in metric_rows if r['condition'] == condition_label]
        rows_for_condition = sorted(rows_for_condition, key=lambda r: r['window_end'])
        xs = [r['window_end'] for r in rows_for_condition]
        ys = [r['delta_pct'] for r in rows_for_condition]
        ax.plot(xs, ys, marker=marker, linewidth=1.9, markersize=5.0,
                color=color, label=condition_label)
        for x, y in zip(xs, ys):
            offset = 2.4 if y >= 0 else -2.4
            va = 'bottom' if y >= 0 else 'top'
            ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, offset),
                        textcoords='offset points', ha='center', va=va,
                        color=color, fontsize=7.2,
                        bbox=dict(boxstyle='round,pad=0.10', facecolor='white',
                                  edgecolor='none', alpha=0.86))
    ax.axhline(0, color='0.45', linestyle='--', linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Delta (%)')
    ax.set_xticks([100, 200, 500, 1000])
    ax.grid(True)
    ax.set_ylim(-20, 130 if metric == 'mse_cum' else 21.5)
    if metric == 'ce_cum':
        ax.set_ylim(-9, 21.5)
    if show_legend:
        ax.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.22))


def _strict_accuracy_rows(cum_rows):
    models = _models_present(cum_rows)
    raw_rows = []
    for model in models:
        model_rows = [r for r in cum_rows if r.get('model') == model]
        seen_files = set()
        acc_vals = []
        for r in model_rows:
            key = (r.get('model'), r.get('file_idx'), r.get('player1_id'), r.get('player2_id'))
            if key in seen_files:
                continue
            seen_files.add(key)
            acc = _to_float(r.get('both_correct'))
            if acc is not None:
                acc_vals.append(acc)
        overall_acc, acc_ci, acc_n = _mean_ci(acc_vals)

        by_w = defaultdict(list)
        for r in model_rows:
            value = _to_float(r.get('strict_cum'))
            w = _to_int(r.get('window_idx'))
            if value is not None and w is not None:
                by_w[w].append(value)
        xs_arr, ys_arr, _ = _mean_ci_arrays(by_w)
        for w, strict_mean in zip(xs_arr, ys_arr):
            raw_rows.append({
                'model': model,
                'model_label': MODEL_LABELS[model],
                'window_idx': int(w),
                'window_end': int(w) * 100,
                'overall_accuracy': overall_acc,
                'overall_accuracy_ci95': acc_ci,
                'overall_accuracy_n': acc_n,
                'cumulative_strict_rate': strict_mean,
                'cumulative_strict_n': len(by_w[int(w)]),
                'difference': strict_mean - overall_acc,
            })
    return raw_rows


def _plot_strict_accuracy_panel(ax, raw_rows, show_legend=False, legend_position='right'):
    models = [m for m in MODEL_ORDER if any(r['model'] == m for r in raw_rows)]
    for model in models:
        rows_for_model = sorted([r for r in raw_rows if r['model'] == model], key=lambda r: r['window_end'])
        if not rows_for_model:
            continue
        xs = [r['window_end'] for r in rows_for_model]
        ys = [r['cumulative_strict_rate'] for r in rows_for_model]
        acc = rows_for_model[0]['overall_accuracy']
        ax.plot(xs, ys, marker='o', linewidth=1.6, markersize=3.5,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.axhline(acc, color=MODEL_COLORS[model], linestyle='--', linewidth=1.15, alpha=0.72)
    ax.set_title('Strict rule match and identity accuracy')
    ax.set_ylabel('Rate')
    ax.set_xlabel('Generated rounds')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(range(100, 1001, 100))
    ax.grid(True)
    if show_legend:
        model_handles, model_labels = ax.get_legend_handles_labels()
        if legend_position == 'inside':
            model_legend = ax.legend(model_handles, model_labels, loc='upper right',
                                     ncol=1, frameon=False, fontsize=7.5)
        elif legend_position == 'top':
            model_legend = ax.legend(model_handles, model_labels, loc='upper center',
                                     ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.20),
                                     fontsize=7.5)
        else:
            model_legend = ax.legend(model_handles, model_labels, loc='upper left',
                                     ncol=1, frameon=False, bbox_to_anchor=(1.02, 1.02),
                                     fontsize=7)
        ax.add_artist(model_legend)
        type_handles = [
            Line2D([0], [0], color='0.35', linestyle='-', linewidth=1.6),
            Line2D([0], [0], color='0.35', linestyle='--', linewidth=1.15),
        ]
        type_labels = ['solid = strict match', 'dashed = identity accuracy']
        if legend_position == 'inside':
            ax.legend(type_handles, type_labels, loc='upper left', ncol=1,
                      frameon=False, fontsize=7.2)
        elif legend_position == 'top':
            ax.legend(type_handles, type_labels, loc='upper center', ncol=2,
                      frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7.2)
        else:
            ax.legend(type_handles, type_labels, loc='upper left', ncol=1,
                      frameon=False, bbox_to_anchor=(1.02, 0.52), fontsize=7)


def _write_rows(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Data -> {path}')


def plot_main_mse_strict_accuracy(cum_rows, outdir):
    """Main Figure 2: MSE delta and strict/identity relationship."""
    delta_rows, condition_specs = _distribution_delta_rows(cum_rows)
    strict_rows = _strict_accuracy_rows(cum_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25))
    _plot_distribution_delta_panel(
        axes[0], delta_rows, condition_specs, 'mse_cum',
        'Distribution MSE delta', show_legend=True,
    )
    _plot_strict_accuracy_panel(axes[1], strict_rows, show_legend=True, legend_position='inside')
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight='bold',
                ha='left', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=0.8, w_pad=1.2)

    path = os.path.join(outdir, 'fig2_mse_strict_accuracy.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')

    _write_rows(
        os.path.join(outdir, 'fig2_mse_delta_raw.csv'),
        ['metric', 'metric_label', 'condition', 'window_end',
         'markov_mean', 'nonmarkov_mean', 'delta_pct', 'markov_n', 'nonmarkov_n'],
        [r for r in delta_rows if r['metric'] == 'mse_cum'],
    )
    _write_rows(
        os.path.join(outdir, 'fig2_strict_identity_raw.csv'),
        ['model', 'model_label', 'window_idx', 'window_end',
         'overall_accuracy', 'overall_accuracy_ci95', 'overall_accuracy_n',
         'cumulative_strict_rate', 'cumulative_strict_n', 'difference'],
        strict_rows,
    )


def plot_main_split_mse_delta(cum_rows, outdir):
    """Main Figure 2-1: standalone MSE delta panel."""
    delta_rows, condition_specs = _distribution_delta_rows(cum_rows)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    _plot_distribution_delta_panel(
        ax, delta_rows, condition_specs, 'mse_cum',
        'Distribution MSE delta', show_legend=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.8)
    path = os.path.join(outdir, 'fig2-1_mse_delta.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_main_split_strict_identity(cum_rows, outdir):
    """Main Figure 2-2: standalone strict rule match and identity accuracy panel."""
    strict_rows = _strict_accuracy_rows(cum_rows)
    fig, ax = plt.subplots(figsize=(7.8, 4.25))
    _plot_strict_accuracy_panel(ax, strict_rows, show_legend=True, legend_position='inside')
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.8)
    path = os.path.join(outdir, 'fig2-2_strict_identity_accuracy.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_ce_delta(cum_rows, outdir):
    """Appendix Figure A1: CE delta separated from main Figure 2."""
    delta_rows, condition_specs = _distribution_delta_rows(cum_rows)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    _plot_distribution_delta_panel(
        ax, delta_rows, condition_specs, 'ce_cum',
        'Distribution CE delta', show_legend=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=0.8)
    path = os.path.join(outdir, 'figA1_distribution_ce_delta.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')
    _write_rows(
        os.path.join(outdir, 'figA1_distribution_ce_delta_raw.csv'),
        ['metric', 'metric_label', 'condition', 'window_end',
         'markov_mean', 'nonmarkov_mean', 'delta_pct', 'markov_n', 'nonmarkov_n'],
        [r for r in delta_rows if r['metric'] == 'ce_cum'],
    )


def plot_appendix_rule_breakdown_xyz(rows, outdir):
    """Appendix: rule-following separated by X/Y/Z Markov identity."""
    markov_rows = [r for r in rows if r.get('markov_player_id') in MARKOV_IDS]
    models = _models_present(markov_rows)
    rules = ['X', 'Y', 'Z']
    metrics = [
        ('overlap_rate', 'Rule overlap', 'Overlap rate'),
        ('strict_exact', 'Strict rule match', 'Strict rate'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    x = np.arange(len(rules))
    width = 0.18
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for offset, model in zip(offsets, models):
            means, cis = [], []
            for rule in rules:
                vals = [
                    _to_float(r.get(metric))
                    for r in markov_rows
                    if r.get('model') == model and r.get('markov_player_id') == rule
                ]
                mean, ci, _ = _mean_ci(vals)
                means.append(mean)
                cis.append(ci)
            ax.bar(x + offset, means, width, yerr=cis, capsize=2,
                   color=MODEL_COLORS[model], edgecolor='black', linewidth=0.4,
                   label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(rules)
        ax.set_xlabel('Markov rule')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight='bold', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.8)
    path = os.path.join(outdir, 'figA10_rule_breakdown_xyz.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_main_strict_cumulative_dynamics(cum_rows, outdir):
    """Main Figure 3: model-level cumulative dynamics for strict rule matching."""
    models = _models_present(cum_rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    for model in models:
        by_w = defaultdict(list)
        for r in cum_rows:
            if r.get('model') != model:
                continue
            value = _to_float(r.get('strict_cum'))
            w = _to_int(r.get('window_idx'))
            if value is not None and w is not None:
                by_w[w].append(value)
        xs_arr, ys_arr, ci_arr = _mean_ci_arrays(by_w)
        round_ends = xs_arr * 100
        ax.plot(round_ends, ys_arr, marker='o', linewidth=1.6, markersize=3.5,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_title('Cumulative strict rule match')
    ax.set_ylabel('Cumulative strict rate')
    ax.set_xlabel('Generated rounds')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(range(100, 1001, 100))
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.8)
    path = os.path.join(outdir, 'figA2_cumulative_strict_rule_match.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_main_accuracy_strict_relationship(cum_rows, outdir):
    """Main Figure 3-2: Figure 3 clone with identification accuracy reference lines."""
    models = _models_present(cum_rows)
    raw_rows = []
    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    for model in models:
        model_rows = [r for r in cum_rows if r.get('model') == model]

        seen_files = set()
        acc_vals = []
        for r in model_rows:
            key = (r.get('model'), r.get('file_idx'), r.get('player1_id'), r.get('player2_id'))
            if key in seen_files:
                continue
            seen_files.add(key)
            acc = _to_float(r.get('both_correct'))
            if acc is not None:
                acc_vals.append(acc)
        overall_acc, acc_ci, acc_n = _mean_ci(acc_vals)

        by_w = defaultdict(list)
        for r in model_rows:
            value = _to_float(r.get('strict_cum'))
            w = _to_int(r.get('window_idx'))
            if value is not None and w is not None:
                by_w[w].append(value)

        xs_arr, ys_arr, _ = _mean_ci_arrays(by_w)
        if len(xs_arr) == 0:
            continue
        round_ends = xs_arr * 100
        ax.plot(round_ends, ys_arr, marker='o', linewidth=1.6, markersize=3.5,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.axhline(overall_acc, color=MODEL_COLORS[model], linestyle='--',
                   linewidth=1.15, alpha=0.72)

        for w, strict_mean in zip(xs_arr, ys_arr):
            raw_rows.append({
                'model': model,
                'model_label': MODEL_LABELS[model],
                'window_idx': int(w),
                'window_end': int(w) * 100,
                'overall_accuracy': overall_acc,
                'overall_accuracy_ci95': acc_ci,
                'overall_accuracy_n': acc_n,
                'cumulative_strict_rate': strict_mean,
                'cumulative_strict_n': len(by_w[int(w)]),
                'difference': strict_mean - overall_acc,
            })

    ax.set_title('Cumulative strict rule match across generated rounds')
    ax.set_ylabel('Rate')
    ax.set_xlabel('Generated rounds')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(range(100, 1001, 100))
    ax.grid(True)

    model_handles, model_labels = ax.get_legend_handles_labels()
    model_legend = fig.legend(
        model_handles,
        model_labels,
        loc='upper center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.add_artist(model_legend)
    type_handles = [
        Line2D([0], [0], color='0.35', linestyle='-', linewidth=1.6),
        Line2D([0], [0], color='0.35', linestyle='--', linewidth=1.15),
    ]
    type_labels = [
        'solid = cumulative strict rule match',
        'dashed = overall identity accuracy',
    ]
    fig.legend(
        type_handles,
        type_labels,
        loc='upper center',
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82], pad=0.8)

    path = os.path.join(outdir, 'fig3-2_accuracy_strict_relationship.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')

    raw_path = os.path.join(outdir, 'fig3-2_accuracy_strict_relationship_raw.csv')
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'model', 'model_label', 'window_idx', 'window_end',
            'overall_accuracy', 'overall_accuracy_ci95', 'overall_accuracy_n',
            'cumulative_strict_rate', 'cumulative_strict_n', 'difference',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f'  Data -> {raw_path}')


def _file_rows(rows):
    seen = set()
    out = []
    for r in rows:
        key = (r.get('model'), r.get('file_idx'))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _binary_counts(file_rows, model, strict=False):
    tp = fp = fn = tn = 0
    for r in file_rows:
        if r.get('model') != model:
            continue
        for true_id, pred_id in [(r.get('player1_id'), r.get('pred_p1_id')), (r.get('player2_id'), r.get('pred_p2_id'))]:
            if not pred_id:
                continue
            true_markov = true_id in MARKOV_IDS
            pred_markov = pred_id in MARKOV_IDS
            exact = pred_id == true_id
            if strict:
                tp += int(true_markov and exact)
                fp += int((not true_markov) and pred_markov)
                fn += int(true_markov and not exact)
                tn += int((not true_markov) and not pred_markov)
            else:
                tp += int(true_markov and pred_markov)
                fp += int((not true_markov) and pred_markov)
                fn += int(true_markov and not pred_markov)
                tn += int((not true_markov) and not pred_markov)
    return tp, fp, fn, tn


def _draw_binary_cm(ax, tp, fp, fn, tn, title):
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    total = mat.sum()
    ax.imshow(mat, cmap='Blues', vmin=0, vmax=max(total, 1))
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val = int(mat[i, j])
            pct = val / total * 100 if total else 0
            ax.text(j, i, f'{labels[i][j]}\n{val}\n{pct:.0f}%', ha='center', va='center',
                    fontsize=7, color='white' if mat[i, j] > total * 0.45 else 'black')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-M', 'M'])
    ax.set_yticklabels(['Non-M', 'M'])
    ax.set_title(title)
    ax.grid(False)


def plot_appendix_binary_confusion(rows, outdir):
    file_rows = _file_rows(rows)
    models = _models_present(file_rows)
    fig, axes = plt.subplots(2, len(models), figsize=(8.0, 3.8))

    for col, model in enumerate(models):
        _draw_binary_cm(axes[0, col], *_binary_counts(file_rows, model, strict=False), MODEL_LABELS[model])
        _draw_binary_cm(axes[1, col], *_binary_counts(file_rows, model, strict=True), MODEL_LABELS[model])
        axes[1, col].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual\nClass')
    axes[1, 0].set_ylabel('Actual\nExact ID')

    fig.tight_layout(pad=0.7)
    path = os.path.join(outdir, 'figA5_binary_confusion.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_full_confusion(rows, outdir):
    file_rows = _file_rows(rows)
    models = _models_present(file_rows)
    id_to_idx = {v: i for i, v in enumerate(VALID_IDS)}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8))
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        mat = np.zeros((len(VALID_IDS), len(VALID_IDS)), dtype=float)
        for r in file_rows:
            if r.get('model') != model:
                continue
            for true_id, pred_id in [(r.get('player1_id'), r.get('pred_p1_id')), (r.get('player2_id'), r.get('pred_p2_id'))]:
                ti = id_to_idx.get(true_id)
                pi = id_to_idx.get(pred_id)
                if ti is not None and pi is not None:
                    mat[ti, pi] += 1

        row_sum = mat.sum(axis=1, keepdims=True)
        pct = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum != 0)
        im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xticks(range(len(VALID_IDS)))
        ax.set_yticks(range(len(VALID_IDS)))
        ax.set_xticklabels(VALID_IDS, fontsize=6)
        ax.set_yticklabels(VALID_IDS, fontsize=6)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.grid(False)
    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.93, wspace=0.28, hspace=0.34)
    cax = fig.add_axes([0.89, 0.22, 0.025, 0.56])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Row-normalized proportion')
    path = os.path.join(outdir, 'figA6_full_identity_confusion.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _strategy_id(x):
    return (x or '').strip().upper()


def _strategy_ce_mse_by_model():
    """Return per-window CE/MSE values for both generated player sides."""
    records = run_analysis.load_all_jsons()
    values = {
        'ce': defaultdict(list),
        'mse': defaultdict(list),
    }

    for rec in records:
        model = rec.get('_model_dir') or rec.get('model')
        if model not in MODEL_ORDER:
            continue
        ctx = rec.get('context', {})
        llm = rec.get('llm_simulation', {})
        side_specs = [
            (
                _strategy_id(rec.get('player1_id')),
                ctx.get('p1_stats', {}),
                llm.get('p1_window_stats', []),
            ),
            (
                _strategy_id(rec.get('player2_id')),
                ctx.get('p2_stats', {}),
                llm.get('p2_window_stats', []),
            ),
        ]
        for strategy, baseline_stats, window_stats in side_specs:
            if strategy not in VALID_IDS:
                continue
            baseline = run_analysis._pct_vec(baseline_stats)
            for ce, mse in run_analysis.ce_mse_by_window(baseline, window_stats):
                values['ce'][(strategy, model)].append(ce)
                values['ce'][(strategy, 'Overall')].append(ce)
                values['mse'][(strategy, model)].append(mse)
                values['mse'][(strategy, 'Overall')].append(mse)
    return values


def _plot_strategy_metric_by_model(metric_values, metric, ylabel, title, path):
    strategies = [s for s in VALID_IDS if any(metric_values.get((s, series)) for series in ['Overall'] + MODEL_ORDER)]
    series_order = ['Overall'] + [m for m in MODEL_ORDER if any(metric_values.get((s, m)) for s in strategies)]
    series_labels = {'Overall': 'Overall', **MODEL_LABELS}
    series_colors = {'Overall': '#777777', **MODEL_COLORS}

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    x = np.arange(len(strategies))
    width = min(0.15, 0.78 / max(len(series_order), 1))
    offsets = (np.arange(len(series_order)) - (len(series_order) - 1) / 2) * width

    for offset, series in zip(offsets, series_order):
        means, cis = [], []
        for strategy in strategies:
            mean, ci, _ = _mean_ci(metric_values.get((strategy, series), []))
            means.append(mean)
            cis.append(ci)
        ax.bar(x + offset, means, width, yerr=cis, capsize=1.8,
               color=series_colors[series], edgecolor='black', linewidth=0.35,
               label=series_labels[series])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('True strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(series_order), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.985, 0.015, f'error bars: 95% CI; {metric.upper()} lower is better',
             ha='right', va='bottom', fontsize=7, color='0.35')
    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=0.8)
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _plot_strategy_overall_summary(values, path):
    strategies = [
        s for s in VALID_IDS
        if values['ce'].get((s, 'Overall')) or values['mse'].get((s, 'Overall'))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    panels = [
        (axes[0], values['ce'], 'Distribution CE by true strategy', 'CE', '#8DA0CB'),
        (axes[1], values['mse'], 'Distribution MSE by true strategy', 'MSE', '#66C2A5'),
    ]

    x = np.arange(len(strategies))
    for ax, metric_values, title, ylabel, color in panels:
        means, cis = [], []
        for strategy in strategies:
            mean, ci, _ = _mean_ci(metric_values.get((strategy, 'Overall'), []))
            means.append(mean)
            cis.append(ci)
        ax.bar(x, means, yerr=cis, capsize=2.2, color=color,
               edgecolor='black', linewidth=0.45)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('True strategy')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)

    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, fontweight='bold',
                ha='left', va='top')
    fig.text(0.985, 0.015, 'error bars: 95% CI; lower is better',
             ha='right', va='bottom', fontsize=7, color='0.35')
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.8)
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_strategy_generation(rows, outdir):
    """Appendix Figure A3: CE/MSE generation quality by true strategy."""
    values = _strategy_ce_mse_by_model()
    _plot_strategy_metric_by_model(
        values['ce'],
        'ce',
        'CE',
        'Distribution CE by true strategy',
        os.path.join(outdir, 'figA7a_strategy_ce_by_model.png'),
    )
    _plot_strategy_metric_by_model(
        values['mse'],
        'mse',
        'MSE',
        'Distribution MSE by true strategy',
        os.path.join(outdir, 'figA7b_strategy_mse_by_model.png'),
    )
    _plot_strategy_overall_summary(
        values,
        os.path.join(outdir, 'figA7c_strategy_overall_summary.png'),
    )


def plot_appendix_per_model_cumulative(cum_rows, outdir):
    """Appendix: per-model cumulative rule diagnostics."""
    models = _models_present(cum_rows)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        for metric, label, color in [
            ('overlap_cum', 'Overlap', MODEL_COLORS[model]),
            ('strict_cum', 'Strict', '0.25'),
        ]:
            by_w = defaultdict(list)
            for r in cum_rows:
                if r.get('model') != model:
                    continue
                value = _to_float(r.get(metric))
                w = _to_int(r.get('window_idx'))
                if value is not None and w is not None:
                    by_w[w].append(value)
            xs, ys, cis = _mean_ci_arrays(by_w)
            if len(xs):
                ax.fill_between(xs, np.maximum(0, ys - cis), np.minimum(1, ys + cis),
                                color=color, alpha=0.12, linewidth=0)
                ax.plot(xs, ys, marker='o', linewidth=1.5, markersize=3,
                        color=color, label=label)
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, 11))
        ax.grid(True)

    for ax in axes[2:]:
        ax.set_xlabel('Window index')
    for ax in axes[::2]:
        ax.set_ylabel('Cumulative rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.985, 0.015, 'shaded bands: 95% CI', ha='right', va='bottom', fontsize=7, color='0.35')
    fig.tight_layout(rect=[0, 0, 1, 0.93], pad=0.8)
    path = os.path.join(outdir, 'figA8_per_model_cumulative_diagnostics.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_distribution_diagnostics(cum_rows, outdir):
    """Appendix: distributional generation diagnostics by model."""
    models = _models_present(cum_rows)
    metrics = [
        ('ce_cum', 'Cumulative CE'),
        ('mse_cum', 'Cumulative MSE'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    for ax, (metric, title) in zip(axes, metrics):
        for model in models:
            by_w = defaultdict(list)
            for r in cum_rows:
                if r.get('model') != model:
                    continue
                value = _to_float(r.get(metric))
                w = _to_int(r.get('window_idx'))
                if value is not None and w is not None:
                    by_w[w].append(value)
            xs, ys, cis = _mean_ci_arrays(by_w)
            if len(xs):
                ax.fill_between(xs, ys - cis, ys + cis, color=MODEL_COLORS[model], alpha=0.12, linewidth=0)
                ax.plot(xs, ys, marker='o', linewidth=1.5, markersize=3,
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel('Window index')
        ax.set_ylabel(title)
        ax.set_xticks(range(1, 11))
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.985, 0.015, 'shaded bands: 95% CI', ha='right', va='bottom', fontsize=7, color='0.35')
    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=0.8)
    path = os.path.join(outdir, 'figA9_distribution_diagnostics.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def main():
    _setup_style()
    rows = _read_csv(METRICS_CSV)
    _ensure_cumulative_csv()
    cum_rows = _read_csv(CUMULATIVE_CSV)

    os.makedirs(MAIN_PNG, exist_ok=True)
    os.makedirs(APPENDIX_PNG, exist_ok=True)

    main_expected = {
        'fig1_rule_following_summary.png',
        'fig2_mse_strict_accuracy.png',
        'fig2-1_mse_delta.png',
        'fig2-2_strict_identity_accuracy.png',
    }
    appendix_expected = {
        'figA1_distribution_ce_delta.png',
        'figA2_cumulative_strict_rule_match.png',
        'figA3_identity_condition_effect.png',
        'figA4_identity_condition_probability_side.png',
        'figA5_binary_confusion.png',
        'figA6_full_identity_confusion.png',
        'figA7a_strategy_ce_by_model.png',
        'figA7b_strategy_mse_by_model.png',
        'figA7c_strategy_overall_summary.png',
        'figA8_per_model_cumulative_diagnostics.png',
        'figA9_distribution_diagnostics.png',
        'figA10_rule_breakdown_xyz.png',
    }

    print('\nGenerating paper main figures ...')
    with mirror_png_to_pdf(MAIN_PNG, MAIN_PDF):
        plot_main_rule_following(rows, MAIN_PNG)
        plot_main_mse_strict_accuracy(cum_rows, MAIN_PNG)
        plot_main_split_mse_delta(cum_rows, MAIN_PNG)
        plot_main_split_strict_identity(cum_rows, MAIN_PNG)
    verify_expected(MAIN_PNG, main_expected)
    verify_expected(MAIN_PDF, expected_pdf_files(main_expected))

    print('\nGenerating paper appendix figures ...')
    with mirror_png_to_pdf(APPENDIX_PNG, APPENDIX_PDF):
        plot_appendix_ce_delta(cum_rows, APPENDIX_PNG)
        plot_main_strict_cumulative_dynamics(cum_rows, APPENDIX_PNG)
        plot_main_identity_condition(cum_rows, APPENDIX_PNG)
        plot_main_identity_condition_nonmarkov(cum_rows, APPENDIX_PNG)
        plot_appendix_binary_confusion(rows, APPENDIX_PNG)
        plot_appendix_full_confusion(rows, APPENDIX_PNG)
        plot_appendix_strategy_generation(rows, APPENDIX_PNG)
        plot_appendix_per_model_cumulative(cum_rows, APPENDIX_PNG)
        plot_appendix_distribution_diagnostics(cum_rows, APPENDIX_PNG)
        plot_appendix_rule_breakdown_xyz(rows, APPENDIX_PNG)
    verify_expected(APPENDIX_PNG, appendix_expected)
    verify_expected(APPENDIX_PDF, expected_pdf_files(appendix_expected))

    print(f'\nDone. Paper plots -> {PAPER_ROOT}')


if __name__ == '__main__':
    main()
