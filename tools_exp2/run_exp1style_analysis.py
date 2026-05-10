"""
Exp2 — Generate exp1-style plots for exp2 data.

Reads xyz_metrics.csv and produces the same plot types as exp1(strategy)/plots/:
  acc, mda, markov_exact, markov_precision, markov_recall, markov_f1,
  ce, brier, tv  (individual line plots)
  confusion_markov/   (2x2 Markov-class confusion matrices)
  confusion_identity/ (strict version — exact identity required for TP)

x-axis: window_idx × 100  (= simulation rounds 100..1000)

Input:  exp2(generation_blind)/analysis_results/xyz_metrics.csv
Output: exp2(generation_blind)/plots_exp1/

Usage:
  python tools_exp2/run_exp1style_analysis.py
"""

import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP2_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
RESULT_ROOT = os.path.join(EXP2_ROOT, 'analysis_results')
PLOT_ROOT   = os.path.join(EXP2_ROOT, 'plots_exp1')

MARKOV_IDS = {'X', 'Y', 'Z'}

METRIC_INFO = {
    'acc':              'ACC',
    'mda':              'MDA',
    'markov_exact':     'Markov Exact',
    'markov_precision': 'Markov Precision',
    'markov_recall':    'Markov Recall',
    'markov_f1':        'Markov F1',
    'ce':               'CE',
    'brier':            'Brier',
    'tv':               'TV',
}

COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v', 'p']


# ── helpers ──────────────────────────────────────────────────────────────────

def to_float(x):
    try: return float(x)
    except Exception: return None


def to_int(x):
    try: return int(x)
    except Exception: return None


def safe_div(a, b):
    return a / b if b > 0 else 0.0


# ── per-sample metrics ────────────────────────────────────────────────────────

def sample_metrics(row):
    """Return per-sample metric dict, or None if row is unparseable."""
    markov_side    = row.get('markov_side', '')
    pred_p1        = row.get('pred_p1_id', '')
    pred_p2        = row.get('pred_p2_id', '')

    if markov_side == 'p1':
        pred_markov    = pred_p1
        pred_nonmarkov = pred_p2
    elif markov_side == 'p2':
        pred_markov    = pred_p2
        pred_nonmarkov = pred_p1
    else:
        return None

    # Type A: Markov class detection (is prediction in {X,Y,Z}?)
    tp_a = 1 if pred_markov    in MARKOV_IDS else 0
    fp_a = 1 if pred_nonmarkov in MARKOV_IDS else 0
    fn_a = 1 - tp_a
    tn_a = 1 - fp_a

    # Type B: exact identity required for TP
    markov_correct = to_int(row.get('markov_correct', ''))
    tp_b = 1 if markov_correct == 1 else 0
    fn_b = 1 - tp_b
    fp_b = fp_a   # non-Markov side unchanged
    tn_b = tn_a

    overlap = to_float(row.get('overlap_rate', ''))

    return dict(
        tp_a=tp_a, fp_a=fp_a, fn_a=fn_a, tn_a=tn_a,
        tp_b=tp_b, fp_b=fp_b, fn_b=fn_b, tn_b=tn_b,
        acc          = to_int(row.get('both_correct', '')),
        mda          = to_int(row.get('markov_id_match', '')),
        markov_exact = overlap,
        ce           = to_float(row.get('ce', '')),
        brier        = to_float(row.get('mse', '')),
        tv           = (1.0 - overlap) if overlap is not None else None,
    )


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows):
    """Group by (model, window_idx) and compute exp1-style metrics."""
    grp = defaultdict(lambda: dict(
        tp_a=0, fp_a=0, fn_a=0, tn_a=0,
        tp_b=0, fp_b=0, fn_b=0, tn_b=0,
        acc_s=0.0, mda_s=0.0, me_s=0.0, ce_s=0.0, br_s=0.0, tv_s=0.0,
        n_acc=0, n_mda=0, n_me=0, n_ce=0, n_br=0, n_tv=0, n=0,
    ))

    for row in rows:
        m = sample_metrics(row)
        if m is None:
            continue
        model = row.get('model', '')
        w     = to_int(row.get('window_idx', ''))
        if w is None:
            continue
        g = grp[(model, w)]
        g['n'] += 1
        for k in ('tp_a', 'fp_a', 'fn_a', 'tn_a', 'tp_b', 'fp_b', 'fn_b', 'tn_b'):
            g[k] += m[k]

        def acc_s(key, sum_k, n_k):
            v = m[key]
            if v is not None:
                g[sum_k] += v
                g[n_k]   += 1

        acc_s('acc',          'acc_s', 'n_acc')
        acc_s('mda',          'mda_s', 'n_mda')
        acc_s('markov_exact', 'me_s',  'n_me')
        acc_s('ce',           'ce_s',  'n_ce')
        acc_s('brier',        'br_s',  'n_br')
        acc_s('tv',           'tv_s',  'n_tv')

    records = []
    for (model, w), g in sorted(grp.items()):
        tp_a, fp_a, fn_a, tn_a = g['tp_a'], g['fp_a'], g['fn_a'], g['tn_a']
        tp_b, fp_b, fn_b, tn_b = g['tp_b'], g['fp_b'], g['fn_b'], g['tn_b']

        prec = safe_div(tp_a, tp_a + fp_a)
        rec  = safe_div(tp_a, tp_a + fn_a)
        f1   = safe_div(2 * prec * rec, prec + rec)

        records.append(dict(
            model         = model,
            rounds        = w * 100,
            window_idx    = w,
            type          = 'overall',
            n             = g['n'],
            acc           = safe_div(g['acc_s'], g['n_acc']) if g['n_acc'] else 0.0,
            mda           = safe_div(g['mda_s'], g['n_mda']) if g['n_mda'] else 0.0,
            markov_exact  = safe_div(g['me_s'],  g['n_me'])  if g['n_me']  else 0.0,
            markov_precision = prec,
            markov_recall    = rec,
            markov_f1        = f1,
            ce    = safe_div(g['ce_s'], g['n_ce']) if g['n_ce'] else 0.0,
            brier = safe_div(g['br_s'], g['n_br']) if g['n_br'] else 0.0,
            tv    = safe_div(g['tv_s'], g['n_tv']) if g['n_tv'] else 0.0,
            markov_tp=tp_a, markov_fp=fp_a, markov_fn=fn_a, markov_tn=tn_a,
            markov_tp_strict=tp_b, markov_fp_strict=fp_b,
            markov_fn_strict=fn_b, markov_tn_strict=tn_b,
        ))

    return records


# ── line plots ────────────────────────────────────────────────────────────────

def plot_metric(records, metric, outdir):
    """One line plot per metric, one line per model — matches exp1 style."""
    by_model = defaultdict(list)
    for r in records:
        by_model[r['model']].append((r['rounds'], r[metric]))

    models  = sorted(by_model)
    all_rounds = sorted({r['rounds'] for r in records})

    plt.figure(figsize=(10, 6))

    for idx, model in enumerate(models):
        pts   = sorted(by_model[model])
        x, y  = zip(*pts) if pts else ([], [])
        color  = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        plt.plot(list(x), list(y),
                 marker=marker, color=color,
                 linewidth=2, markersize=8, label=model, alpha=0.8)

    label = METRIC_INFO.get(metric, metric.upper())
    plt.title(f'{label} vs Rounds', fontsize=14, pad=10)
    plt.xlabel('Rounds (simulation steps seen)', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(all_rounds)
    plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(outdir, f'{metric}_vs_rounds_overall.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Plot -> {path}')


# ── confusion matrices ────────────────────────────────────────────────────────

def _draw_cm(ax, tp, fp, fn, tn, title=''):
    mat   = np.array([[tn, fp], [fn, tp]], dtype=float)
    total = mat.sum()
    ax.imshow(mat, cmap=plt.cm.Blues, vmin=0, vmax=max(total, 1))
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val  = int(mat[i, j])
            pct  = val / total * 100 if total > 0 else 0
            dark = mat[i, j] / max(total, 1) > 0.55
            ax.text(j, i, f"{labels[i][j]}\n{val}\n({pct:.0f}%)",
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if dark else 'black')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Markov', 'Markov'], fontsize=9)
    ax.set_yticklabels(['Non-Markov', 'Markov'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')


def plot_confusion_set(records, tp_key, fp_key, fn_key, tn_key,
                       title_prefix, outdir):
    os.makedirs(outdir, exist_ok=True)
    models     = sorted({r['model'] for r in records})
    all_rounds = sorted({r['rounds'] for r in records})
    n_models   = len(models)

    # per-round figures
    for rounds in all_rounds:
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
        for j, model in enumerate(models):
            hits = [r for r in records if r['model'] == model and r['rounds'] == rounds]
            tp, fp, fn, tn = (hits[0][tp_key], hits[0][fp_key],
                               hits[0][fn_key], hits[0][tn_key]) if hits else (0, 0, 0, 0)
            _draw_cm(axes[0][j], tp, fp, fn, tn, title=model)
        fig.suptitle(f'{title_prefix}\noverall | Rounds = {rounds}', fontsize=13, y=0.97)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path = os.path.join(outdir, f'confusion_overall_rounds{rounds}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Plot -> {path}')

    # aggregated figure
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
    for j, model in enumerate(models):
        m_rows = [r for r in records if r['model'] == model]
        tp = sum(r[tp_key] for r in m_rows)
        fp = sum(r[fp_key] for r in m_rows)
        fn = sum(r[fn_key] for r in m_rows)
        tn = sum(r[tn_key] for r in m_rows)
        _draw_cm(axes[0][j], tp, fp, fn, tn, title=model)
    fig.suptitle(f'{title_prefix}\noverall | All Rounds Aggregated', fontsize=13, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'confusion_overall_all.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Plot -> {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    in_path = os.path.join(RESULT_ROOT, 'xyz_metrics.csv')
    if not os.path.exists(in_path):
        print(f'ERROR: {in_path} not found. Run run_analysis.py first.')
        return

    with open(in_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f'Loaded {len(rows)} rows from {in_path}')

    records = aggregate(rows)
    print(f'Aggregated into {len(records)} (model, window) groups')

    os.makedirs(PLOT_ROOT, exist_ok=True)

    # Line plots
    print('\n-- Line plots --')
    for metric in METRIC_INFO:
        plot_metric(records, metric, PLOT_ROOT)

    # Confusion matrices
    print('\n-- Confusion matrices (class only) --')
    plot_confusion_set(
        records,
        'markov_tp', 'markov_fp', 'markov_fn', 'markov_tn',
        'Markov Detection (class only)',
        os.path.join(PLOT_ROOT, 'confusion_markov'),
    )

    print('\n-- Confusion matrices (strict — exact identity) --')
    plot_confusion_set(
        records,
        'markov_tp_strict', 'markov_fp_strict', 'markov_fn_strict', 'markov_tn_strict',
        'Markov Detection (exact identity required for TP)',
        os.path.join(PLOT_ROOT, 'confusion_identity'),
    )

    print(f'\nDone. Plots -> {PLOT_ROOT}')


if __name__ == '__main__':
    main()
