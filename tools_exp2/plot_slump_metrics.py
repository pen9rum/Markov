import os
import csv
import math
import matplotlib.pyplot as plt


COMBO_COLORS = {
    1: '#1f77b4',  # Type1
    2: '#ff7f0e',  # Type2
    3: '#2ca02c',  # Type3
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_for_model(agg_dict, model, outdir):
    # agg_dict: {(model, combo, window): {'ce_sum':.., 'mse_sum':.., 'exact_sum':.., 'exact_count':.., 'strict_sum':.., 'strict_count':.., 'count':..}}
    ensure_dir(outdir)

    # collect combo types for this model
    combos = sorted({k[1] for k in agg_dict.keys() if k[0] == model})
    if not combos:
        return

    # collect window indices
    windows = sorted({k[2] for k in agg_dict.keys() if k[0] == model})

    # Map combo types to descriptions
    combo_names = {
        1: "Type1: NonMarkov vs NonMarkov",
        2: "Type2: Markov_P1 vs NonMarkov",
        3: "Type3: NonMarkov vs Markov_P2",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for c in combos:
        x = []
        ce_y = []
        mse_y = []
        exact_y = []
        exact_x = []
        strict_y = []
        strict_x = []
        for w in windows:
            key = (model, c, w)
            v = agg_dict.get(key)
            if v and v['count'] > 0:
                x.append(w)
                ce_y.append(v['ce_sum'] / v['count'])
                mse_y.append(v['mse_sum'] / v['count'])
                if v['exact_count'] > 0:
                    exact_x.append(w)
                    exact_y.append(v['exact_sum'] / v['exact_count'])
                if v['strict_count'] > 0:
                    strict_x.append(w)
                    strict_y.append(v['strict_sum'] / v['strict_count'])

        if x:
            label = combo_names.get(c, f'combo_{c}')
            color = COMBO_COLORS.get(c)
            axes[0].plot(x, ce_y, marker='o', label=label, linewidth=2, color=color)
            axes[1].plot(x, mse_y, marker='o', label=label, linewidth=2, color=color)
            if exact_x:
                axes[2].plot(exact_x, exact_y, marker='o', label=label, linewidth=2, color=color)
            if strict_x:
                axes[3].plot(strict_x, strict_y, marker='o', label=label, linewidth=2, color=color)

    axes[0].set_title(f'{model} — Cross-Entropy vs Window', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Window Index (100 rounds per window)', fontsize=10)
    axes[0].set_ylabel('Cross-Entropy', fontsize=10)
    axes[0].set_xticks(windows)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc='best')

    axes[1].set_title(f'{model} — MSE vs Window', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Window Index (100 rounds per window)', fontsize=10)
    axes[1].set_ylabel('Mean Squared Error', fontsize=10)
    axes[1].set_xticks(windows)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc='best')

    axes[2].set_title(f'{model} — Markov Exact Match vs Window', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Window Index (100 rounds per window)', fontsize=10)
    axes[2].set_ylabel('Exact Match Rate', fontsize=10)
    axes[2].set_xticks(windows)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].axhline(1.0 / 3.0, color='purple', linestyle='--', linewidth=1.5, label='Random baseline (1/3)')
    axes[2].grid(True, alpha=0.3)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(fontsize=9, loc='best')

    axes[3].set_title(f'{model} — Markov Strict Exact Match vs Window', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Window Index (100 rounds per window)', fontsize=10)
    axes[3].set_ylabel('Strict Pass Rate', fontsize=10)
    axes[3].set_xticks(windows)
    axes[3].set_ylim(0.0, 1.05)
    axes[3].grid(True, alpha=0.3)
    handles, labels = axes[3].get_legend_handles_labels()
    if handles:
        axes[3].legend(fontsize=9, loc='best')

    plt.tight_layout()
    outpath = os.path.join(outdir, f'{model}_slump.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', outpath)


def main():
    src = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'analysis_results', 'slump_metrics.csv')
    outdir = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'plots_generation', 'slump')
    if not os.path.exists(src):
        print('ERROR: slump_metrics.csv not found')
        return

    agg = {}
    models = set()
    with open(src, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get('model') or '').strip()
            try:
                combo = int(float(row.get('combo_type') or 0))
            except Exception:
                combo = 0
            try:
                window = int(float(row.get('window_idx') or 0))
            except Exception:
                window = 0
            try:
                ce = float(row.get('ce') or 0.0)
            except Exception:
                ce = 0.0
            try:
                mse = float(row.get('mse') or 0.0)
            except Exception:
                mse = 0.0
            exact = row.get('exact_match')
            exact_val = None
            if exact not in (None, ''):
                try:
                    exact_val = float(exact)
                except Exception:
                    exact_val = None
            strict = row.get('strict_exact_match_all')
            strict_val = None
            if strict not in (None, ''):
                try:
                    strict_val = float(strict)
                except Exception:
                    strict_val = None

            key = (model, combo, window)
            if key not in agg:
                agg[key] = {
                    'ce_sum': 0.0,
                    'mse_sum': 0.0,
                    'exact_sum': 0.0,
                    'exact_count': 0,
                    'strict_sum': 0.0,
                    'strict_count': 0,
                    'count': 0,
                }
            agg[key]['ce_sum'] += ce
            agg[key]['mse_sum'] += mse
            if exact_val is not None:
                agg[key]['exact_sum'] += exact_val
                agg[key]['exact_count'] += 1
            if strict_val is not None:
                agg[key]['strict_sum'] += strict_val
                agg[key]['strict_count'] += 1
            agg[key]['count'] += 1
            models.add(model)

    for m in sorted(models):
        plot_for_model(agg, m, outdir)


if __name__ == '__main__':
    main()
