import os
import csv
import matplotlib.pyplot as plt


PLOT_DIR = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'plots_generation', 'slump')
INPUT_CSV = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'analysis_results', 'cumulative_metrics.csv')

MODEL_COLORS = {
    'deepseek-chat': '#1f77b4',
    'deepseek-reasoner': '#ff7f0e',
    'gpt-5': '#2ca02c',
    'gpt-5-mini': '#d62728',
}

MARKER_STYLE = 'o'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def load_rows():
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def markov_identity_match(row):
    side = (row.get('markov_side') or '').strip()
    if side not in ('p1', 'p2'):
        return None
    if side == 'p1':
        return 1 if (row.get('pred_p1_id') or '').strip() == (row.get('player1_id') or '').strip() else 0
    if side == 'p2':
        return 1 if (row.get('pred_p2_id') or '').strip() == (row.get('player2_id') or '').strip() else 0
    return None


def group_avg(rows, model=None, condition=None, metric='strict_cum'):
    by_window = {}
    counts = {}
    for row in rows:
        if model is not None and (row.get('model') or '').strip() != model:
            continue
        if condition is not None:
            cond = condition(row)
            if cond is None or not cond:
                continue
        window = int(float(row.get('window_idx') or 0))
        value = to_float(row.get(metric), None)
        if value is None:
            continue
        by_window[window] = by_window.get(window, 0.0) + value
        counts[window] = counts.get(window, 0) + 1
    xs = sorted(by_window.keys())
    ys = [by_window[x] / counts[x] for x in xs]
    return xs, ys


def plot_markov_metric(rows, metric, title, ylabel, outname, by_model=False):
    if by_model:
        models = ['deepseek-chat', 'deepseek-reasoner', 'gpt-5', 'gpt-5-mini']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for idx, model in enumerate(models):
            ax = axes[idx]
            color = MODEL_COLORS.get(model, '#2E86DE')
            correct_x, correct_y = group_avg(rows, model=model, condition=lambda r: markov_identity_match(r) == 1, metric=metric)
            wrong_x, wrong_y = group_avg(rows, model=model, condition=lambda r: markov_identity_match(r) == 0, metric=metric)
            ax.plot(correct_x, correct_y, marker=MARKER_STYLE, linewidth=2, color=color, label='Identity Correct')
            ax.plot(wrong_x, wrong_y, marker=MARKER_STYLE, linewidth=2, color='#E74C3C', label='Identity Wrong')
            ax.set_title(f'{model} — {title}')
            ax.set_xlabel('Window Index (100 rounds per window)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        outpath = os.path.join(PLOT_DIR, outname)
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('WROTE', outpath)
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        correct_x, correct_y = group_avg(rows, condition=lambda r: markov_identity_match(r) == 1, metric=metric)
        wrong_x, wrong_y = group_avg(rows, condition=lambda r: markov_identity_match(r) == 0, metric=metric)
        ax.plot(correct_x, correct_y, marker=MARKER_STYLE, linewidth=2, color='#2E86DE', label='Identity Correct')
        ax.plot(wrong_x, wrong_y, marker=MARKER_STYLE, linewidth=2, color='#E74C3C', label='Identity Wrong')
        ax.set_title(title)
        ax.set_xlabel('Window Index (100 rounds per window)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        outpath = os.path.join(PLOT_DIR, outname)
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('WROTE', outpath)


def plot_markov_strict(rows, by_model=False):
    plot_markov_metric(
        rows,
        metric='strict_cum',
        title='Markov Player — Cumulative Strict Exact Match',
        ylabel='Cumulative Strict Pass Rate',
        outname='identity_condition_markov_strict_by_model_cumulative.png' if by_model else 'identity_condition_markov_strict_cumulative.png',
        by_model=by_model,
    )


def plot_markov_exact(rows, by_model=False):
    plot_markov_metric(
        rows,
        metric='exact_cum',
        title='Markov Player — Cumulative Exact Match',
        ylabel='Cumulative Exact Match Rate',
        outname='identity_condition_markov_exact_cumulative.png' if not by_model else 'identity_condition_markov_exact_by_model_cumulative.png',
        by_model=by_model,
    )


def main():
    ensure_dir(PLOT_DIR)
    if not os.path.exists(INPUT_CSV):
        print(f'ERROR: {INPUT_CSV} not found')
        return
    rows = load_rows()
    plot_markov_exact(rows, by_model=False)
    plot_markov_exact(rows, by_model=True)
    plot_markov_strict(rows, by_model=False)
    plot_markov_strict(rows, by_model=True)


if __name__ == '__main__':
    main()
