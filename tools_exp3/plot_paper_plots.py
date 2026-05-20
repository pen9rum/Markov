"""
Generate publication-focused Exp2-vs-Exp3 comparison figures.

The Exp3 paper figures are aligned to Exp2 by using only deepseek-reasoner
from Exp2/XYZ as the baseline, then comparing it with Exp3/QRS, TUV, and
XYZ-opp complex Markov rule families.

Output:
  exp3(complex_markov)/paper_plots_exp3/main/png/
  exp3(complex_markov)/paper_plots_exp3/main/pdf/
  exp3(complex_markov)/paper_plots_exp3/appendix/png/
  exp3(complex_markov)/paper_plots_exp3/appendix/pdf/
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools_exp2'))
from plot_output import EXPORT_DPI, expected_pdf_files, mirror_png_to_pdf, split_output_roots, verify_expected


ROOT = os.path.join(os.path.dirname(__file__), '..')
EXP2_RESULT_ROOT = os.path.join(ROOT, 'exp2(generation_blind)', 'analysis_results')
EXP3_ROOT = os.path.join(ROOT, 'exp3(complex_markov)')
EXP3_RESULT_ROOT = os.path.join(EXP3_ROOT, 'analysis_results')
PAPER_ROOT = os.path.join(EXP3_ROOT, 'paper_plots_exp3')
MAIN_ROOT = os.path.join(PAPER_ROOT, 'main')
APPENDIX_ROOT = os.path.join(PAPER_ROOT, 'appendix')
MAIN_PNG, MAIN_PDF = split_output_roots(MAIN_ROOT)
APPENDIX_PNG, APPENDIX_PDF = split_output_roots(APPENDIX_ROOT)

MODEL = 'deepseek-reasoner'
FAMILY_ORDER = ['exp2_xyz', 'qrs', 'xyz_opp', 'tuv']
FAMILY_LABELS = {
    'exp2_xyz': 'XYZ',
    'qrs': 'QRS',
    'tuv': 'TUV',
    'xyz_opp': 'XYZ-opp',
}
MAIN_FAMILY_LABELS = {
    'exp2_xyz': '1st-order\nopp-only',
    'qrs': '1st-order\njoint',
    'xyz_opp': '2nd-order\nopp-only',
    'tuv': '2nd-order\njoint',
}
MAIN_FAMILY_TITLES = {
    'exp2_xyz': 'First-order opponent-only',
    'qrs': 'First-order joint-state',
    'xyz_opp': 'Second-order opponent-only',
    'tuv': 'Second-order joint-state',
}
FAMILY_TITLES = {
    'exp2_xyz': 'XYZ',
    'qrs': 'QRS',
    'tuv': 'TUV',
    'xyz_opp': 'XYZ-opp',
}
VALID_IDS = {
    'exp2_xyz': list('ABCDEFGHIJKLMNOPXYZ'),
    'qrs': list('ABCDEFGHIJKLMNOPQRS'),
    'tuv': list('ABCDEFGHIJKLMNOPTUV'),
    'xyz_opp': list('ABCDEFGHIJKLMNOPxyz'),
}
COND_COLORS = {
    'correct': '#4477AA',
    'wrong': '#EE6677',
    'overall': '#777777',
}
FAMILY_COLORS = {
    'exp2_xyz': '#4477AA',
    'qrs': '#228833',
    'tuv': '#CCBB44',
    'xyz_opp': '#EE6677',
}


def _setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 12,
        'figure.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run the analysis scripts first.')
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


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


def _load_family_rows():
    rows_by_family = {}
    exp2_rows = _read_csv(os.path.join(EXP2_RESULT_ROOT, 'xyz_metrics.csv'))
    rows_by_family['exp2_xyz'] = [
        {**r, 'family': 'exp2_xyz'}
        for r in exp2_rows
        if r.get('model') == MODEL
    ]
    for family in ['qrs', 'tuv', 'xyz_opp']:
        path = os.path.join(EXP3_RESULT_ROOT, f'{family}_metrics.csv')
        rows_by_family[family] = [
            {**r, 'family': family}
            for r in _read_csv(path)
            if r.get('model') == MODEL
        ]
    return rows_by_family


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


def _condition_values(rows, metric, condition):
    vals = []
    for r in rows:
        value = _to_float(r.get(metric))
        if value is None:
            continue
        cond = _to_int(r.get('markov_id_match'))
        if condition == 'overall' and cond is not None:
            vals.append(value)
        elif condition == 'correct' and cond == 1:
            vals.append(value)
        elif condition == 'wrong' and cond == 0:
            vals.append(value)
    return vals


def plot_main_rule_following(rows_by_family, outdir):
    metric_specs = [
        ('overlap_rate', 'Rule overlap', 'Overlap rate', 1 / 3),
        ('strict_exact', 'Strict rule match', 'Strict rate', None),
    ]
    cond_specs = [
        ('correct', 'Correct', COND_COLORS['correct']),
        ('wrong', 'Wrong', COND_COLORS['wrong']),
        ('overall', 'Overall', COND_COLORS['overall']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), sharey=True)
    x = np.arange(len(FAMILY_ORDER))
    width = 0.23
    offsets = (np.arange(len(cond_specs)) - 1) * width

    for ax, (metric, title, ylabel, baseline) in zip(axes, metric_specs):
        for offset, (cond, label, color) in zip(offsets, cond_specs):
            means, cis = [], []
            for family in FAMILY_ORDER:
                rows = [r for r in rows_by_family[family] if _to_float(r.get(metric)) is not None]
                mean, ci, _ = _mean_ci(_condition_values(rows, metric, cond))
                means.append(mean)
                cis.append(ci)
            ax.bar(x + offset, means, width, yerr=cis, capsize=2.2,
                   color=color, edgecolor='black', linewidth=0.45, label=label)
        if baseline is not None:
            ax.axhline(baseline, color='0.35', linestyle='--', linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([MAIN_FAMILY_LABELS[f] for f in FAMILY_ORDER], fontsize=9.5)
        ax.tick_params(axis='x', pad=7)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='y', labelleft=True)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='0.35', linestyle='--', linewidth=1.0))
    labels.append('random baseline')
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0.08, 1, 0.91], pad=0.8, w_pad=1.8)
    path = os.path.join(outdir, 'fig1_exp2_exp3_rule_following_comparison.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _rate(rows, col):
    vals = [_to_float(r.get(col)) for r in rows]
    return _mean_ci(vals)


def plot_main_identification(rows_by_family, outdir):
    metric_specs = [
        ('both_correct', 'Overall identity', '#777777'),
        ('markov_correct', 'Markov identity', '#4477AA'),
        ('nonmarkov_correct', 'Non-Markov identity', '#EE6677'),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(FAMILY_ORDER))
    width = 0.23
    offsets = (np.arange(len(metric_specs)) - 1) * width

    for offset, (col, label, color) in zip(offsets, metric_specs):
        means, cis = [], []
        for family in FAMILY_ORDER:
            f_rows = _file_rows(rows_by_family[family])
            if col == 'markov_correct':
                f_rows = [r for r in f_rows if _to_int(r.get('combo_type')) in (2, 3)]
            mean, ci, _ = _rate(f_rows, col)
            means.append(mean)
            cis.append(ci)
        ax.bar(
            x + offset, means, width, yerr=cis, capsize=2.4,
            color=color, edgecolor='black', linewidth=0.8, label=label,
            error_kw={'ecolor': 'black', 'elinewidth': 0.9, 'capthick': 0.9},
        )

    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([MAIN_FAMILY_LABELS[f] for f in FAMILY_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis='both', colors='black', width=0.9)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.grid(axis='y', color='0.55', alpha=0.35, linewidth=0.7)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.8)
    path = os.path.join(outdir, 'fig2_exp2_exp3_identification_comparison.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _cumulative_strict_by_file(rows):
    groups = defaultdict(list)
    for r in rows:
        if _to_float(r.get('strict_exact')) is None:
            continue
        key = (
            r.get('model'),
            r.get('file_idx'),
            r.get('player1_id'),
            r.get('player2_id'),
            r.get('pred_p1_id'),
            r.get('pred_p2_id'),
        )
        groups[key].append(r)

    by_w = defaultdict(list)
    for grp in groups.values():
        ok = True
        for r in sorted(grp, key=lambda x: _to_int(x.get('window_idx')) or 0):
            strict = _to_float(r.get('strict_exact'))
            if strict is None:
                continue
            if strict != 1.0:
                ok = False
            w = _to_int(r.get('window_idx'))
            if w is not None:
                by_w[w].append(1.0 if ok else 0.0)
    return by_w


def _cumulative_metric_by_file(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        if _to_float(r.get(metric)) is None:
            continue
        key = (
            r.get('model'),
            r.get('file_idx'),
            r.get('player1_id'),
            r.get('player2_id'),
            r.get('pred_p1_id'),
            r.get('pred_p2_id'),
        )
        groups[key].append(r)

    by_w = defaultdict(list)
    for grp in groups.values():
        running_sum = 0.0
        running_n = 0
        strict_ok = True
        for r in sorted(grp, key=lambda x: _to_int(x.get('window_idx')) or 0):
            value = _to_float(r.get(metric))
            if value is None:
                continue
            if metric == 'strict_exact':
                if value != 1.0:
                    strict_ok = False
                out_value = 1.0 if strict_ok else 0.0
            else:
                running_sum += value
                running_n += 1
                out_value = running_sum / running_n
            w = _to_int(r.get('window_idx'))
            if w is not None:
                by_w[w].append(out_value)
    return by_w


def plot_main_cumulative_strict(rows_by_family, outdir):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for family in FAMILY_ORDER:
        by_w = _cumulative_strict_by_file(rows_by_family[family])
        xs = sorted(by_w)
        means, cis = [], []
        for w in xs:
            mean, ci, _ = _mean_ci(by_w[w])
            means.append(mean)
            cis.append(ci)
        xs_arr = np.array(xs)
        ys_arr = np.array(means)
        ci_arr = np.array(cis)
        round_ends = xs_arr * 100
        ax.plot(round_ends, ys_arr, marker='o', linewidth=1.6, markersize=3.5,
                color=FAMILY_COLORS[family], label=MAIN_FAMILY_TITLES[family])

    ax.set_ylabel('Cumulative strict rate')
    ax.set_xlabel('Generated rounds')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(range(100, 1001, 100))
    ax.grid(True)
    ax.legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(rect=[0, 0, 1, 0.84], pad=0.8)
    path = os.path.join(outdir, 'fig3_exp2_exp3_cumulative_strict_dynamics.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def _draw_binary_cm(ax, rows, family, strict=False):
    markov_ids = set(VALID_IDS[family][-3:])
    if family == 'xyz_opp':
        markov_ids = {'x', 'y', 'z'}
    tp = fp = fn = tn = 0
    for r in _file_rows(rows):
        for true_id, pred_id in [(r.get('player1_id'), r.get('pred_p1_id')), (r.get('player2_id'), r.get('pred_p2_id'))]:
            if not pred_id:
                continue
            true_markov = true_id in markov_ids
            pred_markov = pred_id in markov_ids
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
    ax.grid(False)


def plot_appendix_binary_confusion(rows_by_family, outdir):
    fig, axes = plt.subplots(2, len(FAMILY_ORDER), figsize=(8.0, 3.8))
    for col, family in enumerate(FAMILY_ORDER):
        _draw_binary_cm(axes[0, col], rows_by_family[family], family, strict=False)
        _draw_binary_cm(axes[1, col], rows_by_family[family], family, strict=True)
        axes[0, col].set_title(FAMILY_TITLES[family])
        axes[1, col].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual\nClass')
    axes[1, 0].set_ylabel('Actual\nExact ID')
    fig.tight_layout(pad=0.7)
    path = os.path.join(outdir, 'figA1_exp2_exp3_binary_confusion.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_full_confusion(rows_by_family, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8))
    axes = axes.flatten()
    im = None

    for ax, family in zip(axes, FAMILY_ORDER):
        ids = VALID_IDS[family]
        id_to_idx = {v: i for i, v in enumerate(ids)}
        mat = np.zeros((len(ids), len(ids)), dtype=float)
        for r in _file_rows(rows_by_family[family]):
            for true_id, pred_id in [(r.get('player1_id'), r.get('pred_p1_id')), (r.get('player2_id'), r.get('pred_p2_id'))]:
                ti = id_to_idx.get(true_id)
                pi = id_to_idx.get(pred_id)
                if ti is not None and pi is not None:
                    mat[ti, pi] += 1
        row_sum = mat.sum(axis=1, keepdims=True)
        pct = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum != 0)
        im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(FAMILY_TITLES[family])
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids, fontsize=6)
        ax.set_yticklabels(ids, fontsize=6)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.grid(False)

    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.93, wspace=0.30, hspace=0.34)
    cax = fig.add_axes([0.89, 0.22, 0.025, 0.56])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Row-normalized proportion')
    path = os.path.join(outdir, 'figA2_exp2_exp3_full_identity_confusion.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_distribution_diagnostics(rows_by_family, outdir):
    metric_specs = [
        ('ce', 'CE', None),
        ('mse', 'MSE', None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, (metric, ylabel, ylim) in zip(axes, metric_specs):
        means, cis = [], []
        for family in FAMILY_ORDER:
            vals = [_to_float(r.get(metric)) for r in rows_by_family[family]]
            mean, ci, _ = _mean_ci(vals)
            means.append(mean)
            cis.append(ci)
        x = np.arange(len(FAMILY_ORDER))
        ax.bar(x, means, yerr=cis, capsize=2.2,
               color=[FAMILY_COLORS[f] for f in FAMILY_ORDER],
               edgecolor='black', linewidth=0.45)
        ax.set_title(f'Distribution {ylabel}')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER])
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, fontweight='bold', ha='left', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.8)
    path = os.path.join(outdir, 'figA3_exp2_exp3_distribution_diagnostics.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_cumulative_diagnostics(rows_by_family, outdir):
    dist_specs = [
        ('ce', 'Cumulative CE', None),
        ('mse', 'Cumulative MSE', None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    for ax, (metric, title, ylim) in zip(axes, dist_specs):
        for family in FAMILY_ORDER:
            by_w = _cumulative_metric_by_file(rows_by_family[family], metric)
            xs = sorted(by_w)
            means, cis = [], []
            for w in xs:
                mean, ci, _ = _mean_ci(by_w[w])
                means.append(mean)
                cis.append(ci)
            xs_arr = np.array(xs)
            ys_arr = np.array(means)
            ci_arr = np.array(cis)
            if len(xs_arr):
                ax.plot(xs_arr, ys_arr, marker='o', linewidth=1.4, markersize=3,
                        color=FAMILY_COLORS[family], label=FAMILY_TITLES[family])
                ax.fill_between(xs_arr, ys_arr - ci_arr, ys_arr + ci_arr,
                                color=FAMILY_COLORS[family], alpha=0.10, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel('Window index')
        ax.set_ylabel(title)
        ax.set_xticks(range(1, 11))
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, fontweight='bold', ha='left', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=0.8)
    path = os.path.join(outdir, 'figA4a_exp2_exp3_cumulative_distribution.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for family in FAMILY_ORDER:
        by_w = _cumulative_metric_by_file(rows_by_family[family], 'overlap_rate')
        xs = sorted(by_w)
        means, cis = [], []
        for w in xs:
            mean, ci, _ = _mean_ci(by_w[w])
            means.append(mean)
            cis.append(ci)
        xs_arr = np.array(xs)
        ys_arr = np.array(means)
        ci_arr = np.array(cis)
        if len(xs_arr):
            ax.plot(xs_arr, ys_arr, marker='o', linewidth=1.4, markersize=3,
                    color=FAMILY_COLORS[family], label=FAMILY_TITLES[family])
            ax.fill_between(xs_arr, np.maximum(0, ys_arr - ci_arr), np.minimum(1, ys_arr + ci_arr),
                            color=FAMILY_COLORS[family], alpha=0.10, linewidth=0)
    ax.set_title('Cumulative overlap')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Cumulative overlap')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=0.8)
    path = os.path.join(outdir, 'figA4b_exp2_exp3_cumulative_overlap.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_identity_condition_strict(rows_by_family, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, family in zip(axes, FAMILY_ORDER):
        for cond, label, color in [
            ('correct', 'Identity correct', COND_COLORS['correct']),
            ('wrong', 'Identity wrong', COND_COLORS['wrong']),
        ]:
            rows = [
                r for r in rows_by_family[family]
                if _to_float(r.get('strict_exact')) is not None
                and ((cond == 'correct' and _to_int(r.get('markov_id_match')) == 1)
                     or (cond == 'wrong' and _to_int(r.get('markov_id_match')) == 0))
            ]
            by_w = _cumulative_metric_by_file(rows, 'strict_exact')
            xs = sorted(by_w)
            means, cis = [], []
            for w in xs:
                mean, ci, _ = _mean_ci(by_w[w])
                means.append(mean)
                cis.append(ci)
            xs_arr = np.array(xs)
            ys_arr = np.array(means)
            ci_arr = np.array(cis)
            if len(xs_arr):
                ax.plot(xs_arr, ys_arr, marker='o', linewidth=1.4, markersize=3,
                        color=color, label=label, zorder=5, clip_on=False)
                ax.fill_between(xs_arr, np.maximum(0, ys_arr - ci_arr), np.minimum(1, ys_arr + ci_arr),
                                color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.set_title(FAMILY_TITLES[family])
        ax.set_xticks(range(1, 11))
        ax.set_ylim(-0.04, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.spines['bottom'].set_zorder(0)
        ax.spines['left'].set_zorder(0)
        ax.grid(True)

    for ax in axes[2:]:
        ax.set_xlabel('Window index')
    for ax in axes[::2]:
        ax.set_ylabel('Cumulative strict rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0.07, 1, 0.93], pad=0.8)
    path = os.path.join(outdir, 'figA5_exp2_exp3_identity_condition_strict.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_rule_breakdown(rows_by_family, outdir):
    fig, axes = plt.subplots(2, len(FAMILY_ORDER), figsize=(8.4, 4.8), sharey='row')
    metric_specs = [
        ('overlap_rate', 'Overlap rate', (0, 1.05), 0),
        ('strict_exact', 'Strict rate', (0, 1.05), 1),
    ]

    for col, family in enumerate(FAMILY_ORDER):
        markov_ids = VALID_IDS[family][-3:]
        for metric, ylabel, ylim, row_idx in metric_specs:
            ax = axes[row_idx, col]
            means, cis = [], []
            for markov_id in markov_ids:
                vals = [
                    _to_float(r.get(metric))
                    for r in rows_by_family[family]
                    if r.get('markov_player_id') == markov_id
                ]
                mean, ci, _ = _mean_ci(vals)
                means.append(mean)
                cis.append(ci)
            x = np.arange(len(markov_ids))
            ax.bar(x, means, yerr=cis, capsize=2.0, color=FAMILY_COLORS[family],
                   edgecolor='black', linewidth=0.45)
            ax.set_xticks(x)
            ax.set_xticklabels(markov_ids)
            ax.set_ylim(*ylim)
            ax.grid(axis='y')
            ax.grid(axis='x', visible=False)
            if row_idx == 0:
                ax.set_title(FAMILY_TITLES[family])
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 1:
                ax.set_xlabel('Markov rule')

    fig.tight_layout(rect=[0, 0.06, 1, 0.96], pad=0.8)
    path = os.path.join(outdir, 'figA6_exp2_exp3_rule_breakdown.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def plot_appendix_identity_condition_distribution(rows_by_family, outdir):
    metric_specs = [
        ('ce', 'CE'),
        ('mse', 'MSE'),
    ]
    cond_specs = [
        (1, 'Identity correct', COND_COLORS['correct']),
        (0, 'Identity wrong', COND_COLORS['wrong']),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    x = np.arange(len(FAMILY_ORDER))
    width = 0.32
    offsets = [-width / 2, width / 2]

    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for offset, (cond_value, label, color) in zip(offsets, cond_specs):
            means, cis = [], []
            for family in FAMILY_ORDER:
                vals = [
                    _to_float(r.get(metric))
                    for r in rows_by_family[family]
                    if _to_int(r.get('both_correct')) == cond_value
                ]
                mean, ci, _ = _mean_ci(vals)
                means.append(mean)
                cis.append(ci)
            ax.bar(x + offset, means, width, yerr=cis, capsize=2.2,
                   color=color, edgecolor='black', linewidth=0.45, label=label)
        ax.set_title(f'Distribution {ylabel}')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER])
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, fontweight='bold', ha='left', va='top')
    fig.tight_layout(rect=[0, 0.07, 1, 0.92], pad=0.8)
    path = os.path.join(outdir, 'figA7_exp2_exp3_identity_condition_distribution.png')
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot -> {path}')


def main():
    _setup_style()
    rows_by_family = _load_family_rows()

    os.makedirs(MAIN_PNG, exist_ok=True)
    os.makedirs(APPENDIX_PNG, exist_ok=True)
    main_expected = {
        'fig1_exp2_exp3_rule_following_comparison.png',
        'fig2_exp2_exp3_identification_comparison.png',
        'fig3_exp2_exp3_cumulative_strict_dynamics.png',
    }
    appendix_expected = {
        'figA1_exp2_exp3_binary_confusion.png',
        'figA2_exp2_exp3_full_identity_confusion.png',
        'figA3_exp2_exp3_distribution_diagnostics.png',
        'figA4a_exp2_exp3_cumulative_distribution.png',
        'figA4b_exp2_exp3_cumulative_overlap.png',
        'figA5_exp2_exp3_identity_condition_strict.png',
        'figA6_exp2_exp3_rule_breakdown.png',
        'figA7_exp2_exp3_identity_condition_distribution.png',
    }

    print('\nGenerating Exp2-vs-Exp3 main figures ...')
    with mirror_png_to_pdf(MAIN_PNG, MAIN_PDF):
        plot_main_rule_following(rows_by_family, MAIN_PNG)
        plot_main_identification(rows_by_family, MAIN_PNG)
        plot_main_cumulative_strict(rows_by_family, MAIN_PNG)
    verify_expected(MAIN_PNG, main_expected)
    verify_expected(MAIN_PDF, expected_pdf_files(main_expected))

    print('\nGenerating Exp2-vs-Exp3 appendix figures ...')
    with mirror_png_to_pdf(APPENDIX_PNG, APPENDIX_PDF):
        plot_appendix_binary_confusion(rows_by_family, APPENDIX_PNG)
        plot_appendix_full_confusion(rows_by_family, APPENDIX_PNG)
        plot_appendix_distribution_diagnostics(rows_by_family, APPENDIX_PNG)
        plot_appendix_cumulative_diagnostics(rows_by_family, APPENDIX_PNG)
        plot_appendix_identity_condition_strict(rows_by_family, APPENDIX_PNG)
        plot_appendix_rule_breakdown(rows_by_family, APPENDIX_PNG)
        plot_appendix_identity_condition_distribution(rows_by_family, APPENDIX_PNG)
    verify_expected(APPENDIX_PNG, appendix_expected)
    verify_expected(APPENDIX_PDF, expected_pdf_files(appendix_expected))

    print(f'\nDone. Exp3 paper plots -> {PAPER_ROOT}')


if __name__ == '__main__':
    main()
