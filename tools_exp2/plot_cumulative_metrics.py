import os
import csv
import matplotlib.pyplot as plt


COMBO_COLORS = {
    1: '#1f77b4',
    2: '#ff7f0e',
    3: '#2ca02c',
}

MODEL_COLORS = {
    'deepseek-chat': '#1f77b4',
    'deepseek-reasoner': '#ff7f0e',
    'gpt-5': '#2ca02c',
    'gpt-5-mini': '#d62728',
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def load_rows(src):
    with open(src, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def grouped_files(rows, model=None):
    files = {}
    for row in rows:
        if model is not None and (row.get('model') or '').strip() != model:
            continue
        fp = row.get('file')
        if fp and fp not in files:
            files[fp] = row
    return list(files.values())


def model_names(rows):
    return sorted({(r.get('model') or '').strip() for r in rows if (r.get('model') or '').strip()})


def max_window_for_model(rows, model):
    max_w = 0
    for r in rows:
        if (r.get('model') or '').strip() != model:
            continue
        max_w = max(max_w, int(float(r.get('window_idx') or 0)))
    return max_w


def cumulative_series(rows, model, metric):
    # Average over files for each window_idx
    by_window = {}
    counts = {}
    for r in rows:
        if (r.get('model') or '').strip() != model:
            continue
        w = int(float(r.get('window_idx') or 0))
        v = to_float(r.get(metric), None)
        if v is None:
            continue
        by_window[w] = by_window.get(w, 0.0) + v
        counts[w] = counts.get(w, 0) + 1
    xs = sorted(by_window.keys())
    ys = [by_window[x] / counts[x] for x in xs]
    return xs, ys


def overall_cumulative_series(rows, metric, filter_fn=None):
    by_window = {}
    counts = {}
    for r in rows:
        if filter_fn is not None and not filter_fn(r):
            continue
        w = int(float(r.get('window_idx') or 0))
        v = to_float(r.get(metric), None)
        if v is None:
            continue
        by_window[w] = by_window.get(w, 0.0) + v
        counts[w] = counts.get(w, 0) + 1
    xs = sorted(by_window.keys())
    ys = [by_window[x] / counts[x] for x in xs]
    return xs, ys


def plot_model(rows, model, outdir):
    # Distribution metrics: Type1 uses averaged distribution, but all rows already in cumulative_metrics.csv
    # We use file-level averages by window.
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    combo_names = {
        1: 'Type1: NonMarkov vs NonMarkov',
        2: 'Type2: Markov_P1 vs NonMarkov',
        3: 'Type3: NonMarkov vs Markov_P2',
    }

    # For per-model plots, aggregate by combo type.
    for combo in [1, 2, 3]:
        combo_rows = [r for r in rows if (r.get('model') or '').strip() == model and int(float(r.get('combo_type') or 0)) == combo]
        if not combo_rows:
            continue

        xs_ce, ys_ce = cumulative_series(combo_rows, model, 'ce_cum')
        xs_mse, ys_mse = cumulative_series(combo_rows, model, 'mse_cum')
        xs_exact, ys_exact = cumulative_series(combo_rows, model, 'exact_cum')
        xs_strict, ys_strict = cumulative_series(combo_rows, model, 'strict_cum')

        label = combo_names.get(combo, f'combo_{combo}')
        color = COMBO_COLORS.get(combo)
        if xs_ce:
            axes[0].plot(xs_ce, ys_ce, marker='o', linewidth=2, label=label, color=color)
        if xs_mse:
            axes[1].plot(xs_mse, ys_mse, marker='o', linewidth=2, label=label, color=color)
        if xs_exact:
            axes[2].plot(xs_exact, ys_exact, marker='o', linewidth=2, label=label, color=color)
        if xs_strict:
            axes[3].plot(xs_strict, ys_strict, marker='o', linewidth=2, label=label, color=color)

    axes[0].set_title(f'{model} — Cumulative CE')
    axes[0].set_xlabel('Window Index (100 rounds per window)')
    axes[0].set_ylabel('CE')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title(f'{model} — Cumulative MSE')
    axes[1].set_xlabel('Window Index (100 rounds per window)')
    axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].set_title(f'{model} — Cumulative Markov Exact Match')
    axes[2].set_xlabel('Window Index (100 rounds per window)')
    axes[2].set_ylabel('Exact Match Rate')
    axes[2].set_ylim(0.0, 1.05)
    axes[2].axhline(1.0 / 3.0, color='purple', linestyle='--', linewidth=1.5, label='Random baseline (1/3)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

    axes[3].set_title(f'{model} — Cumulative Markov Strict Exact Match')
    axes[3].set_xlabel('Window Index (100 rounds per window)')
    axes[3].set_ylabel('Strict Pass Rate')
    axes[3].set_ylim(0.0, 1.05)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(outdir, f'{model}_cumulative.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)


def plot_overall(rows, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    models = model_names(rows)
    metric_specs = [
        ('ce_cum', 'Overall Cumulative CE', 'CE', 0),
        ('mse_cum', 'Overall Cumulative MSE', 'MSE', 1),
        ('exact_cum', 'Overall Cumulative Markov Exact Match', 'Exact Match Rate', 2),
        ('strict_cum', 'Overall Cumulative Markov Strict Exact Match', 'Strict Pass Rate', 3),
    ]

    for metric, title, ylabel, ax_idx in metric_specs:
        ax = axes[ax_idx]
        for model in models:
            model_rows = [r for r in rows if (r.get('model') or '').strip() == model]
            xs, ys = overall_cumulative_series(model_rows, metric)
            if not xs:
                continue
            ax.plot(xs, ys, marker='o', linewidth=2, label=model, color=MODEL_COLORS.get(model))
        ax.set_title(title)
        ax.set_xlabel('Window Index (100 rounds per window)')
        ax.set_ylabel(ylabel)
        if ax_idx >= 2:
            ax.set_ylim(0.0, 1.05)
        if ax_idx == 2:
            ax.axhline(1.0 / 3.0, color='purple', linestyle='--', linewidth=1.5, label='Random baseline (1/3)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'model_comparison_overall_cumulative.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)


def main():
    root = os.getcwd()
    src = os.path.join(root, 'exp2(generation_blind)', 'analysis_results', 'cumulative_metrics.csv')
    outdir = os.path.join(root, 'exp2(generation_blind)', 'plots_generation', 'slump')
    ensure_dir(outdir)

    if not os.path.exists(src):
        print('ERROR: cumulative_metrics.csv not found')
        return

    rows = load_rows(src)
    models = model_names(rows)
    for m in models:
        plot_model(rows, m, outdir)
    plot_overall(rows, outdir)


if __name__ == '__main__':
    main()
