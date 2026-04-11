import os
import csv
import matplotlib.pyplot as plt


MODEL_COLORS = {
    'deepseek-chat': '#1f77b4',
    'deepseek-reasoner': '#ff7f0e',
    'gpt-5': '#2ca02c',
    'gpt-5-mini': '#d62728',
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    src = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'analysis_results', 'slump_metrics.csv')
    outdir = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'plots_generation', 'slump')
    ensure_dir(outdir)

    if not os.path.exists(src):
        print('ERROR: slump_metrics.csv not found')
        return

    # Aggregate by (model, window)
    agg = {}
    windows = set()
    models = set()

    with open(src, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get('model') or '').strip()
            if not model:
                continue

            try:
                window = int(float(row.get('window_idx') or 0))
            except Exception:
                continue

            try:
                ce = float(row.get('ce') or 0.0)
            except Exception:
                ce = 0.0

            try:
                mse = float(row.get('mse') or 0.0)
            except Exception:
                mse = 0.0

            exact_raw = row.get('exact_match')
            exact = None
            if exact_raw not in (None, ''):
                try:
                    exact = float(exact_raw)
                except Exception:
                    exact = None
            strict_raw = row.get('strict_exact_match_all')
            strict = None
            if strict_raw not in (None, ''):
                try:
                    strict = float(strict_raw)
                except Exception:
                    strict = None

            key = (model, window)
            if key not in agg:
                agg[key] = {
                    'ce_sum': 0.0, 'mse_sum': 0.0, 'n': 0,
                    'exact_sum': 0.0, 'exact_n': 0,
                    'strict_sum': 0.0, 'strict_n': 0,
                }

            agg[key]['ce_sum'] += ce
            agg[key]['mse_sum'] += mse
            agg[key]['n'] += 1
            if exact is not None:
                agg[key]['exact_sum'] += exact
                agg[key]['exact_n'] += 1
            if strict is not None:
                agg[key]['strict_sum'] += strict
                agg[key]['strict_n'] += 1

            windows.add(window)
            models.add(model)

    if not models:
        print('ERROR: no rows found in slump_metrics.csv')
        return

    windows = sorted(windows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for model in sorted(models):
        color = MODEL_COLORS.get(model)
        x_ce, y_ce = [], []
        x_mse, y_mse = [], []
        x_exact, y_exact = [], []
        x_strict, y_strict = [], []

        for w in windows:
            key = (model, w)
            v = agg.get(key)
            if not v or v['n'] == 0:
                continue

            x_ce.append(w)
            y_ce.append(v['ce_sum'] / v['n'])
            x_mse.append(w)
            y_mse.append(v['mse_sum'] / v['n'])

            if v['exact_n'] > 0:
                x_exact.append(w)
                y_exact.append(v['exact_sum'] / v['exact_n'])
            if v['strict_n'] > 0:
                x_strict.append(w)
                y_strict.append(v['strict_sum'] / v['strict_n'])

        if x_ce:
            axes[0].plot(x_ce, y_ce, marker='o', linewidth=2, label=model, color=color)
        if x_mse:
            axes[1].plot(x_mse, y_mse, marker='o', linewidth=2, label=model, color=color)
        if x_exact:
            axes[2].plot(x_exact, y_exact, marker='o', linewidth=2, label=model, color=color)
        if x_strict:
            axes[3].plot(x_strict, y_strict, marker='o', linewidth=2, label=model, color=color)

    axes[0].set_title('Model Comparison — CE', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Window Index (100 rounds per window)')
    axes[0].set_ylabel('Cross-Entropy')
    axes[0].set_xticks(windows)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title('Model Comparison — MSE', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Window Index (100 rounds per window)')
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].set_xticks(windows)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].set_title('Model Comparison — Markov Exact Match', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Window Index (100 rounds per window)')
    axes[2].set_ylabel('Exact Match Rate')
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(windows)
    axes[2].axhline(1.0 / 3.0, color='purple', linestyle='--', linewidth=1.5, label='Random baseline (1/3)')
    axes[2].grid(True, alpha=0.3)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(fontsize=9)

    axes[3].set_title('Model Comparison — Markov Strict Exact Match', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Window Index (100 rounds per window)')
    axes[3].set_ylabel('Strict Pass Rate')
    axes[3].set_ylim(0.0, 1.05)
    axes[3].set_xticks(windows)
    axes[3].grid(True, alpha=0.3)
    handles, labels = axes[3].get_legend_handles_labels()
    if handles:
        axes[3].legend(fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'model_comparison_all.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('WROTE', outpath)


if __name__ == '__main__':
    main()
