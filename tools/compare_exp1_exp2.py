"""
Compare exp1 (rounds=1000) vs exp2 (window_idx=10, 1000-round context) identification metrics.
Outputs bar charts to exp1+2/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MARKOV_IDS = {'X', 'Y', 'Z'}
MODELS = ['deepseek-chat', 'deepseek-reasoner', 'gpt-5', 'gpt-5-mini']
MODEL_LABELS = {
    'deepseek-chat': 'deepseek-chat',
    'deepseek-reasoner': 'deepseek-reasoner',
    'gpt-5': 'gpt-5',
    'gpt-5-mini': 'gpt-5-mini',
}

ROOT = os.path.join(os.path.dirname(__file__), '..')
EXP1_CSV = os.path.join(ROOT, 'exp1(strategy)', 'metrics_export.csv')
EXP2_CSV = os.path.join(ROOT, 'exp2(generation_blind)', 'analysis_results', 'xyz_metrics.csv')
OUT_DIR  = os.path.join(ROOT, 'exp1+2')
os.makedirs(OUT_DIR, exist_ok=True)


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

        # ACC: both_correct across type1 + type2
        all_bc = pd.concat([t1['both_correct'], t2['both_correct']])
        acc = all_bc.mean()

        # Markov detection confusion matrix
        # TP: type2 where markov_id_match == 1
        tp = int(t2['markov_id_match'].sum())
        fn = len(t2) - tp

        # FP: type1 where model predicted a Markov ID
        def pred_has_markov(row):
            return (str(row['pred_p1_id']) in MARKOV_IDS or
                    str(row['pred_p2_id']) in MARKOV_IDS)
        fp = int(t1.apply(pred_has_markov, axis=1).sum())
        tn = len(t1) - fp

        total = tp + fp + fn + tn
        mda = (tp + tn) / total if total > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall    = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)

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


def plot_comparison(exp1, exp2, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(MODELS))
    w = 0.35
    labels = [MODEL_LABELS[m] for m in MODELS]

    v1 = [exp1.loc[m, metric] if m in exp1.index else np.nan for m in MODELS]
    v2 = [exp2.loc[m, metric] if m in exp2.index else np.nan for m in MODELS]

    bars1 = ax.bar(x - w/2, v1, w, label='Exp1 (strategy blind)', color='steelblue')
    bars2 = ax.bar(x + w/2, v2, w, label='Exp2 (generation blind)', color='tomato')

    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def main():
    exp1 = load_exp1(MODELS)
    exp2 = compute_exp2_metrics(MODELS)

    print('\n=== Exp1 (rounds=1000, overall) ===')
    print(exp1[['acc','mda','markov_precision','markov_recall','markov_f1']].round(3))
    print('\n=== Exp2 (window_idx=10 = 1000 rounds) ===')
    print(exp2[['acc','mda','markov_precision','markov_recall','markov_f1']].round(3))

    metrics = [
        ('acc',              'Accuracy',          'Overall Accuracy (both players correct)',          'compare_acc.png'),
        ('mda',              'MDA',               'Markov Detection Accuracy',                        'compare_mda.png'),
        ('markov_precision', 'Precision',         'Markov Detection Precision',                       'compare_markov_precision.png'),
        ('markov_recall',    'Recall',            'Markov Detection Recall',                          'compare_markov_recall.png'),
        ('markov_f1',        'F1',                'Markov Detection F1',                              'compare_markov_f1.png'),
    ]

    for col, ylabel, title, fname in metrics:
        plot_comparison(exp1, exp2, col, ylabel, title, fname)

    # Combined plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, (col, ylabel, title, _) in zip(axes, metrics):
        x = np.arange(len(MODELS))
        w = 0.35
        v1 = [exp1.loc[m, col] if m in exp1.index else np.nan for m in MODELS]
        v2 = [exp2.loc[m, col] if m in exp2.index else np.nan for m in MODELS]
        ax.bar(x - w/2, v1, w, label='Exp1', color='steelblue')
        ax.bar(x + w/2, v2, w, label='Exp2', color='tomato')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, fontsize=8)
        ax.set_title(ylabel, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Exp1 vs Exp2 Identification @ 1000 rounds', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'compare_all.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

    # Save CSV
    exp1_out = exp1[['acc','mda','markov_precision','markov_recall','markov_f1']].copy()
    exp1_out.insert(0, 'experiment', 'exp1')
    exp2_out = exp2[['acc','mda','markov_precision','markov_recall','markov_f1']].copy()
    exp2_out.insert(0, 'experiment', 'exp2')
    combined = pd.concat([exp1_out, exp2_out]).reset_index()
    csv_path = os.path.join(OUT_DIR, 'comparison_metrics.csv')
    combined.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
