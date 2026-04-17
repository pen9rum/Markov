import os
import csv
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def avg(vals):
    return sum(vals) / len(vals) if vals else None


def annotate_bars(ax, bars, values, counts, y_top=None):
    if y_top is None:
        y_top = max(values) if values else 0.0
    if y_top <= 0:
        offset = 0.01
    else:
        offset = max(y_top * 0.03, 0.01)
    for rect, v, n in zip(bars, values, counts):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        ax.text(x, y + offset, f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=8)


def _dist_identity_match(row):
    """Return 1/0 for distribution-player identity correctness, or None if unavailable."""
    dist_id = (row.get('dist_player_id') or '').strip()
    p1_id = (row.get('player1_id') or '').strip()
    p2_id = (row.get('player2_id') or '').strip()
    pred_p1 = (row.get('pred_p1_id') or '').strip()
    pred_p2 = (row.get('pred_p2_id') or '').strip()

    if not dist_id:
        return None
    if dist_id == p1_id:
        return 1 if pred_p1 == p1_id else 0
    if dist_id == p2_id:
        return 1 if pred_p2 == p2_id else 0
    return None


def _markov_identity_match(row):
    side = (row.get('markov_side') or '').strip()
    if side not in ('p1', 'p2'):
        return None
    return 1 if to_float(row.get('markov_id_match'), 0.0) == 1.0 else 0


def plot_distribution(slump_csv, outdir):
    rows = []
    with open(slump_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Overall aggregation (window-level)
    ce_correct, ce_wrong = [], []
    mse_correct, mse_wrong = [], []

    # By-model aggregation (window-level)
    model_stats = {}

    for row in rows:
        model = (row.get('model') or '').strip()
        if not model:
            continue

        m = _dist_identity_match(row)
        if m is None:
            continue

        ce = to_float(row.get('ce'), None)
        mse = to_float(row.get('mse'), None)

        if model not in model_stats:
            model_stats[model] = {
                'ce_c': [], 'ce_w': [],
                'mse_c': [], 'mse_w': [],
            }

        if m == 1:
            if ce is not None:
                ce_correct.append(ce)
                model_stats[model]['ce_c'].append(ce)
            if mse is not None:
                mse_correct.append(mse)
                model_stats[model]['mse_c'].append(mse)
        else:
            if ce is not None:
                ce_wrong.append(ce)
                model_stats[model]['ce_w'].append(ce)
            if mse is not None:
                mse_wrong.append(mse)
                model_stats[model]['mse_w'].append(mse)

    # Figure 1: overall distribution metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = ['Identity Correct', 'Identity Wrong']

    ce_vals = [avg(ce_correct) or 0.0, avg(ce_wrong) or 0.0]
    mse_vals = [avg(mse_correct) or 0.0, avg(mse_wrong) or 0.0]
    ce_ns = [len(ce_correct), len(ce_wrong)]
    mse_ns = [len(mse_correct), len(mse_wrong)]

    bars0 = axes[0].bar(groups, ce_vals, color=['#2E86DE', '#E74C3C'])
    axes[0].set_title('Distribution Player — CE')
    axes[0].set_ylabel('Cross-Entropy (lower is better)')
    axes[0].grid(axis='y', alpha=0.3)

    bars1 = axes[1].bar(groups, mse_vals, color=['#2E86DE', '#E74C3C'])
    axes[1].set_title('Distribution Player — MSE')
    axes[1].set_ylabel('MSE (lower is better)')
    axes[1].grid(axis='y', alpha=0.3)

    annotate_bars(axes[0], bars0, ce_vals, ce_ns)
    annotate_bars(axes[1], bars1, mse_vals, mse_ns)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'identity_condition_distribution.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)

    # Figure 2: by-model distribution metrics
    models = sorted(model_stats.keys())
    if not models:
        return

    x = list(range(len(models)))
    width = 0.35

    ce_c = [avg(model_stats[m]['ce_c']) or 0.0 for m in models]
    ce_w = [avg(model_stats[m]['ce_w']) or 0.0 for m in models]
    mse_c = [avg(model_stats[m]['mse_c']) or 0.0 for m in models]
    mse_w = [avg(model_stats[m]['mse_w']) or 0.0 for m in models]
    ce_c_n = [len(model_stats[m]['ce_c']) for m in models]
    ce_w_n = [len(model_stats[m]['ce_w']) for m in models]
    mse_c_n = [len(model_stats[m]['mse_c']) for m in models]
    mse_w_n = [len(model_stats[m]['mse_w']) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    bars00 = axes[0].bar([i - width / 2 for i in x], ce_c, width=width, label='Identity Correct', color='#2E86DE')
    bars01 = axes[0].bar([i + width / 2 for i in x], ce_w, width=width, label='Identity Wrong', color='#E74C3C')
    axes[0].set_title('Distribution CE by Model')
    axes[0].set_ylabel('Cross-Entropy (lower is better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    bars10 = axes[1].bar([i - width / 2 for i in x], mse_c, width=width, label='Identity Correct', color='#2E86DE')
    bars11 = axes[1].bar([i + width / 2 for i in x], mse_w, width=width, label='Identity Wrong', color='#E74C3C')
    axes[1].set_title('Distribution MSE by Model')
    axes[1].set_ylabel('MSE (lower is better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    annotate_bars(axes[0], bars00, ce_c, ce_c_n)
    annotate_bars(axes[0], bars01, ce_w, ce_w_n)
    annotate_bars(axes[1], bars10, mse_c, mse_c_n)
    annotate_bars(axes[1], bars11, mse_w, mse_w_n)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'identity_condition_distribution_by_model.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)


def plot_markov(slump_csv, outdir):
    rows = []
    with open(slump_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Overall (window-level for exact, file-level for strict)
    exact_correct, exact_wrong = [], []

    # file-level dedup for strict metric
    file_rows = {}
    for r in rows:
        fp = r.get('file')
        if fp and fp not in file_rows:
            file_rows[fp] = r

    strict_correct, strict_wrong = [], []

    # by-model stats
    model_stats = {}

    for row in rows:
        model = (row.get('model') or '').strip()
        if not model:
            continue
        m = _markov_identity_match(row)
        if m is None:
            continue

        if model not in model_stats:
            model_stats[model] = {
                'exact_c': [], 'exact_w': [],
                'strict_c': [], 'strict_w': [],
            }

        exact = row.get('exact_match')
        if exact not in (None, ''):
            ev = to_float(exact, None)
            if ev is not None:
                if m == 1:
                    exact_correct.append(ev)
                    model_stats[model]['exact_c'].append(ev)
                else:
                    exact_wrong.append(ev)
                    model_stats[model]['exact_w'].append(ev)

    for row in file_rows.values():
        model = (row.get('model') or '').strip()
        if not model:
            continue
        m = _markov_identity_match(row)
        if m is None:
            continue
        strict = row.get('strict_exact_match_all')
        if strict in (None, ''):
            continue
        sv = to_float(strict, None)
        if sv is None:
            continue
        if model not in model_stats:
            model_stats[model] = {
                'exact_c': [], 'exact_w': [],
                'strict_c': [], 'strict_w': [],
            }
        if m == 1:
            strict_correct.append(sv)
            model_stats[model]['strict_c'].append(sv)
        else:
            strict_wrong.append(sv)
            model_stats[model]['strict_w'].append(sv)

    # Figure 3: overall markov metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = ['Identity Correct', 'Identity Wrong']

    exact_vals = [avg(exact_correct) or 0.0, avg(exact_wrong) or 0.0]
    strict_vals = [avg(strict_correct) or 0.0, avg(strict_wrong) or 0.0]
    exact_ns = [len(exact_correct), len(exact_wrong)]
    strict_ns = [len(strict_correct), len(strict_wrong)]

    bars0 = axes[0].bar(groups, exact_vals, color=['#2E86DE', '#E74C3C'])
    axes[0].set_title('Markov Player — Window Exact Match')
    axes[0].set_ylabel('Rate')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].axhline(1.0 / 3.0, color='purple', linestyle='--', linewidth=1.5, label='Random baseline (1/3)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    bars1 = axes[1].bar(groups, strict_vals, color=['#2E86DE', '#E74C3C'])
    axes[1].set_title('Markov Player — Strict Exact Match (All-or-Nothing)')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)

    annotate_bars(axes[0], bars0, exact_vals, exact_ns, y_top=1.0)
    annotate_bars(axes[1], bars1, strict_vals, strict_ns, y_top=1.0)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'identity_condition_markov.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)

    # Figure 4: by-model markov metrics
    models = sorted(model_stats.keys())
    if not models:
        return

    x = list(range(len(models)))
    width = 0.35

    exact_c = [avg(model_stats[m]['exact_c']) or 0.0 for m in models]
    exact_w = [avg(model_stats[m]['exact_w']) or 0.0 for m in models]
    strict_c = [avg(model_stats[m]['strict_c']) or 0.0 for m in models]
    strict_w = [avg(model_stats[m]['strict_w']) or 0.0 for m in models]
    exact_c_n = [len(model_stats[m]['exact_c']) for m in models]
    exact_w_n = [len(model_stats[m]['exact_w']) for m in models]
    strict_c_n = [len(model_stats[m]['strict_c']) for m in models]
    strict_w_n = [len(model_stats[m]['strict_w']) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    bars00 = axes[0].bar([i - width / 2 for i in x], exact_c, width=width, label='Identity Correct', color='#2E86DE')
    bars01 = axes[0].bar([i + width / 2 for i in x], exact_w, width=width, label='Identity Wrong', color='#E74C3C')
    axes[0].set_title('Markov Window Exact Match by Model')
    axes[0].set_ylabel('Rate')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].axhline(1.0 / 3.0, color='purple', linestyle='--', linewidth=1.5, label='Random baseline (1/3)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    bars10 = axes[1].bar([i - width / 2 for i in x], strict_c, width=width, label='Identity Correct', color='#2E86DE')
    bars11 = axes[1].bar([i + width / 2 for i in x], strict_w, width=width, label='Identity Wrong', color='#E74C3C')
    axes[1].set_title('Markov Strict Exact Match by Model')
    axes[1].set_ylabel('Rate')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    annotate_bars(axes[0], bars00, exact_c, exact_c_n, y_top=1.0)
    annotate_bars(axes[0], bars01, exact_w, exact_w_n, y_top=1.0)
    annotate_bars(axes[1], bars10, strict_c, strict_c_n, y_top=1.0)
    annotate_bars(axes[1], bars11, strict_w, strict_w_n, y_top=1.0)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'identity_condition_markov_by_model.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)


def main():
    root = os.getcwd()
    slump_csv = os.path.join(root, 'exp2(generation_blind)', 'analysis_results', 'slump_metrics.csv')
    outdir = os.path.join(root, 'exp2(generation_blind)', 'plots_generation', 'slump')
    ensure_dir(outdir)

    if not os.path.exists(slump_csv):
        print('ERROR: slump_metrics.csv not found')
        return

    plot_distribution(slump_csv, outdir)
    plot_markov(slump_csv, outdir)


if __name__ == '__main__':
    main()
