"""
Compare exp1 (rounds=1000) vs exp2 (window_idx=10, 1000-round context)
identification metrics.

Outputs matching PNG and PDF bar charts to exp1+2/.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd


MARKOV_IDS = {'X', 'Y', 'Z'}
MODELS = ['deepseek-chat', 'deepseek-reasoner', 'gpt-5', 'gpt-5-mini']
MODEL_LABELS = {
    'deepseek-chat': 'deepseek-chat',
    'deepseek-reasoner': 'deepseek-reasoner',
    'gpt-5': 'gpt-5',
    'gpt-5-mini': 'gpt-5-mini',
}
EXPORT_DPI = 400

ROOT = os.path.join(os.path.dirname(__file__), '..')
EXP1_CSV = os.path.join(ROOT, 'exp1(strategy)', 'metrics_export.csv')
EXP2_CSV = os.path.join(ROOT, 'exp2(generation_blind)', 'analysis_results', 'xyz_metrics.csv')
OUT_DIR = os.path.join(ROOT, 'exp1+2')
os.makedirs(OUT_DIR, exist_ok=True)


plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'medium',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'axes.edgecolor': 'black',
    'axes.titlesize': 16,
    'axes.titleweight': 'semibold',
    'axes.labelsize': 15,
    'axes.labelweight': 'semibold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.fontsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.color': '#8a8a8a',
    'grid.linewidth': 0.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def load_exp1(models):
    df = pd.read_csv(EXP1_CSV)
    df = df[(df['rounds'] == 1000) & (df['type'] == 'overall')]
    df = df[df['model'].isin(models)].set_index('model')
    return df


def compute_exp2_metrics(models):
    df = pd.read_csv(EXP2_CSV)
    df = df[df['window_idx'] == 10]

    rows = []
    for model in models:
        m = df[df['model'] == model]
        t1 = m[m['combo_type'] == 1]
        t2 = m[m['combo_type'] == 2]

        all_bc = pd.concat([t1['both_correct'], t2['both_correct']])
        acc = all_bc.mean()

        tp = int(t2['markov_id_match'].sum())
        fn = len(t2) - tp

        def pred_has_markov(row):
            return (
                str(row['pred_p1_id']) in MARKOV_IDS
                or str(row['pred_p2_id']) in MARKOV_IDS
            )

        fp = int(t1.apply(pred_has_markov, axis=1).sum())
        tn = len(t1) - fp

        total = tp + fp + fn + tn
        mda = (tp + tn) / total if total > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )

        rows.append({
            'model': model,
            'acc': acc,
            'mda': mda,
            'markov_precision': precision,
            'markov_recall': recall,
            'markov_f1': f1,
            'samples_t1': len(t1),
            'samples_t2': len(t2),
        })

    return pd.DataFrame(rows).set_index('model')


def save_figure(fig, png_name):
    png_path = os.path.join(OUT_DIR, png_name)
    pdf_path = os.path.splitext(png_path)[0] + '.pdf'
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=EXPORT_DPI, bbox_inches='tight')
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')


def strengthen_axis_text(ax):
    ax.title.set_color('black')
    ax.title.set_fontweight('semibold')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_fontweight('semibold')
    ax.yaxis.label.set_fontweight('semibold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
        label.set_fontweight('medium')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.9)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color('black')
            text.set_fontweight('medium')


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            txt = ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f'{h:.2f}',
                ha='center',
                va='bottom',
                fontsize=13,
                fontweight='semibold',
                color='black',
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.2, foreground='white'),
                path_effects.Normal(),
            ])


def plot_comparison(exp1, exp2, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    x = np.arange(len(MODELS))
    w = 0.35
    labels = [MODEL_LABELS[m] for m in MODELS]

    v1 = [exp1.loc[m, metric] if m in exp1.index else np.nan for m in MODELS]
    v2 = [exp2.loc[m, metric] if m in exp2.index else np.nan for m in MODELS]

    bars1 = ax.bar(
        x - w / 2, v1, w, label='Exp1 (strategy blind)',
        color='steelblue', edgecolor='black', linewidth=0.65,
    )
    bars2 = ax.bar(
        x + w / 2, v2, w, label='Exp2 (generation blind)',
        color='tomato', edgecolor='black', linewidth=0.65,
    )
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False)
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    strengthen_axis_text(ax)

    fig.tight_layout()
    save_figure(fig, fname)
    plt.close(fig)


def main():
    exp1 = load_exp1(MODELS)
    exp2 = compute_exp2_metrics(MODELS)

    print('\n=== Exp1 (rounds=1000, overall) ===')
    print(exp1[['acc', 'mda', 'markov_precision', 'markov_recall', 'markov_f1']].round(3))
    print('\n=== Exp2 (window_idx=10 = 1000 rounds) ===')
    print(exp2[['acc', 'mda', 'markov_precision', 'markov_recall', 'markov_f1']].round(3))

    metrics = [
        ('acc', 'Accuracy', 'Overall Accuracy (both players correct)', 'compare_acc.png'),
        ('mda', 'MDA', 'Markov Detection Accuracy', 'compare_mda.png'),
        ('markov_precision', 'Precision', 'Markov Detection Precision', 'compare_markov_precision.png'),
        ('markov_recall', 'Recall', 'Markov Detection Recall', 'compare_markov_recall.png'),
        ('markov_f1', 'F1', 'Markov Detection F1', 'compare_markov_f1.png'),
    ]

    for col, ylabel, title, fname in metrics:
        plot_comparison(exp1, exp2, col, ylabel, title, fname)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)
    for ax, (col, ylabel, _, _) in zip(axes, metrics):
        x = np.arange(len(MODELS))
        w = 0.35
        v1 = [exp1.loc[m, col] if m in exp1.index else np.nan for m in MODELS]
        v2 = [exp2.loc[m, col] if m in exp2.index else np.nan for m in MODELS]
        ax.bar(x - w / 2, v1, w, label='Exp1', color='steelblue',
               edgecolor='black', linewidth=0.55)
        ax.bar(x + w / 2, v2, w, label='Exp2', color='tomato',
               edgecolor='black', linewidth=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, fontsize=13)
        ax.set_title(ylabel, fontsize=15, fontweight='semibold', color='black')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)
        strengthen_axis_text(ax)
    axes[0].set_ylabel('Score')
    axes[0].legend(frameon=False, fontsize=12)
    strengthen_axis_text(axes[0])
    fig.suptitle('Exp1 vs Exp2 Identification @ 1000 rounds',
                 fontsize=18, fontweight='semibold', color='black')
    fig.tight_layout()
    save_figure(fig, 'compare_all.png')
    plt.close(fig)

    exp1_out = exp1[['acc', 'mda', 'markov_precision', 'markov_recall', 'markov_f1']].copy()
    exp1_out.insert(0, 'experiment', 'exp1')
    exp2_out = exp2[['acc', 'mda', 'markov_precision', 'markov_recall', 'markov_f1']].copy()
    exp2_out.insert(0, 'experiment', 'exp2')
    combined = pd.concat([exp1_out, exp2_out]).reset_index()
    csv_path = os.path.join(OUT_DIR, 'comparison_metrics.csv')
    combined.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
