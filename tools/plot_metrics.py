"""
绘制评估指标随trajectory长度变化的趋势图
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
import os
import math
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


DEFAULT_METRICS = [
    'acc', 'mda', 'markov_exact', 'markov_precision', 'markov_recall', 'markov_f1',
    'tv', 'wr_gap', 'ce', 'brier', 'evloss', 'union'
]


def resolve_metrics(metrics):
    """Normalize metrics input; support 'all'."""
    if metrics is None:
        return DEFAULT_METRICS.copy()

    normalized = [m.strip().lower() for m in metrics if m and str(m).strip()]
    if not normalized or 'all' in normalized:
        return DEFAULT_METRICS.copy()

    return normalized


def plot_metrics_by_rounds(csv_file: str,
                           metrics: list = None,
                           models: list = None,
                           exp_type: str = None,
                           output_dir: str = None,
                           show: bool = True):
    """
    绘制指标随rounds变化的趋势图
    
    Args:
        csv_file: CSV文件路径
        metrics: 要绘制的指标列表，默认所有指标
        models: 要包含的模型列表，默认所有模型
        exp_type: 实验类型过滤 (overall/type1/type2)，默认overall
        output_dir: 输出目录，默认为plots/
        show: 是否显示图表
    """
    # 读取CSV
    df = pd.read_csv(csv_file)
    
    # 默认值
    metrics = resolve_metrics(metrics)
    
    if exp_type is None:
        exp_type = 'overall'
    
    # 过滤数据
    df_filtered = df[df['type'] == exp_type].copy()
    
    if models:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    # 获取所有模型和rounds
    all_models = df_filtered['model'].unique()
    all_rounds = sorted(df_filtered['rounds'].unique())
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metric display names (English only)
    metric_info = {
        'acc': 'ACC',
        'mda': 'MDA',
        'markov_exact': 'Markov Exact',
        'markov_precision': 'Markov Precision',
        'markov_recall': 'Markov Recall',
        'markov_f1': 'Markov F1',
        'tv': 'TV',
        'wr_gap': 'WR Gap',
        'ce': 'CE',
        'brier': 'Brier',
        'evloss': 'EVLoss',
        'union': 'Union'
    }
    
    # 颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 为每个指标绘制单独的图
    for metric in metrics:
        if metric not in df_filtered.columns:
            print(f"警告: 指标 '{metric}' 不存在于数据中，跳过")
            continue
        
        plt.figure(figsize=(10, 6))
        
        # 为每个模型绘制一条线
        for idx, model in enumerate(all_models):
            model_data = df_filtered[df_filtered['model'] == model]
            
            # 按rounds排序并提取数据
            x = []
            y = []
            for rounds in all_rounds:
                round_data = model_data[model_data['rounds'] == rounds]
                if not round_data.empty:
                    x.append(rounds)
                    y.append(round_data[metric].values[0])
            
            # 绘制线条
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            plt.plot(x, y, 
                    marker=marker, 
                    color=color,
                    linewidth=2,
                    markersize=8,
                    label=model,
                    alpha=0.8)
        
        # Title and labels
        title_name = metric_info.get(metric, metric.upper())
        plt.title(f'{title_name} vs Rounds', fontsize=14, pad=10)
        plt.xlabel('Rounds', fontsize=12)
        plt.ylabel(title_name, fontsize=12)
        
        # 设置网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 设置x轴刻度
        plt.xticks(all_rounds)
        
        # 添加图例
        plt.legend(loc='best', framealpha=0.9)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        filename = f"{metric}_vs_rounds_{exp_type}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ 保存图表: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_multiple_metrics(csv_file: str,
                         metrics: list = None,
                         models: list = None,
                         exp_type: str = None,
                         output_dir: str = None,
                         show: bool = True,
                         layout: tuple = None):
    """
    在一张图上绘制多个指标的子图
    
    Args:
        csv_file: CSV文件路径
        metrics: 要绘制的指标列表
        models: 要包含的模型列表
        exp_type: 实验类型过滤
        output_dir: 输出目录
        show: 是否显示图表
        layout: 子图布局 (rows, cols)，默认自动计算
    """
    # 读取CSV
    df = pd.read_csv(csv_file)
    
    # 默认值
    metrics = resolve_metrics(metrics)
    
    if exp_type is None:
        exp_type = 'overall'
    
    # 过滤数据
    df_filtered = df[df['type'] == exp_type].copy()
    
    if models:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    # 获取所有模型和rounds
    all_models = df_filtered['model'].unique()
    all_rounds = sorted(df_filtered['rounds'].unique())
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算子图布局
    n_metrics = len(metrics)
    if layout is None:
        if n_metrics <= 2:
            rows, cols = 1, n_metrics
        elif n_metrics <= 4:
            rows, cols = 2, 2
        else:
            cols = 3
            rows = math.ceil(n_metrics / cols)
    else:
        rows, cols = layout
    
    # Metric display names (English only)
    metric_info = {
        'acc': 'ACC',
        'mda': 'MDA',
        'markov_exact': 'Markov Exact',
        'markov_precision': 'Markov Precision',
        'markov_recall': 'Markov Recall',
        'markov_f1': 'Markov F1',
        'tv': 'TV',
        'wr_gap': 'WR Gap',
        'ce': 'CE',
        'brier': 'Brier',
        'evloss': 'EVLoss',
        'union': 'Union'
    }
    
    # 颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 为每个指标绘制子图
    for idx, metric in enumerate(metrics):
        if metric not in df_filtered.columns:
            print(f"警告: 指标 '{metric}' 不存在，跳过")
            continue
        
        ax = axes[idx]
        
        # 为每个模型绘制线条
        for model_idx, model in enumerate(all_models):
            model_data = df_filtered[df_filtered['model'] == model]
            
            x = []
            y = []
            for rounds in all_rounds:
                round_data = model_data[model_data['rounds'] == rounds]
                if not round_data.empty:
                    x.append(rounds)
                    y.append(round_data[metric].values[0])
            
            color = colors[model_idx % len(colors)]
            marker = markers[model_idx % len(markers)]
            ax.plot(x, y,
                   marker=marker,
                   color=color,
                   linewidth=2,
                   markersize=6,
                   label=model,
                   alpha=0.8)
        
        # Title and labels
        title = metric_info.get(metric, metric.upper())
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel('Rounds', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(all_rounds)
        
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    # Main title (lowered)
    fig.suptitle(f'Metrics vs Rounds ({exp_type})', fontsize=16, y=0.97)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 保存
    filename = f"all_metrics_vs_rounds_{exp_type}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ 保存组合图表: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()


def _draw_confusion_axes(ax, tp, fp, fn, tn, title='', fontsize=11):
    """Draw a single 2x2 confusion matrix into ax."""
    matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
    total = matrix.sum()
    cmap = plt.cm.Blues
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(total, 1))
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val = int(matrix[i, j])
            pct = val / total * 100 if total > 0 else 0
            brightness = matrix[i, j] / max(total, 1)
            color = 'white' if brightness > 0.55 else 'black'
            ax.text(j, i,
                    f"{labels[i][j]}\n{val}\n({pct:.0f}%)",
                    ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Markov', 'Markov'], fontsize=9)
    ax.set_yticklabels(['Non-Markov', 'Markov'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')


def _save_confusion_set(df_f, all_models, all_rounds, exp_type,
                         tp_col, fp_col, fn_col, tn_col,
                         title_prefix, output_dir, show):
    """
    Produce 5 figures (4 per-round + 1 aggregated) for one set of TP/FP/FN/TN columns.
    Each figure has 1 row × n_models columns.
    """
    n_models = len(all_models)
    os.makedirs(output_dir, exist_ok=True)

    def get_counts(row_data):
        if row_data.empty:
            return 0, 0, 0, 0
        rd = row_data.iloc[0]
        return (int(rd.get(tp_col) or 0),
                int(rd.get(fp_col) or 0),
                int(rd.get(fn_col) or 0),
                int(rd.get(tn_col) or 0))

    # ── 4 per-round figures ──────────────────────────────────────────────
    for rounds in all_rounds:
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
        for col_j, model in enumerate(all_models):
            ax = axes[0][col_j]
            row_data = df_f[(df_f['model'] == model) & (df_f['rounds'] == rounds)]
            tp, fp, fn, tn = get_counts(row_data)
            _draw_confusion_axes(ax, tp, fp, fn, tn, title=model)
        fig.suptitle(
            f'{title_prefix}\n{exp_type} | Rounds = {rounds}',
            fontsize=13, y=0.97
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"confusion_{exp_type}_rounds{rounds}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f"✓ 保存: {fpath}")
        plt.show() if show else plt.close()

    # ── 1 aggregated figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
    for col_j, model in enumerate(all_models):
        ax = axes[0][col_j]
        md = df_f[df_f['model'] == model]
        tp = int(md[tp_col].fillna(0).sum())
        fp = int(md[fp_col].fillna(0).sum())
        fn = int(md[fn_col].fillna(0).sum())
        tn = int(md[tn_col].fillna(0).sum())
        _draw_confusion_axes(ax, tp, fp, fn, tn, title=model)
    fig.suptitle(
        f'{title_prefix}\n{exp_type} | All Rounds Aggregated',
        fontsize=13, y=0.97
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"confusion_{exp_type}_all.png"
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {fpath}")
    plt.show() if show else plt.close()


def plot_confusion_matrices(csv_file: str,
                             models: list = None,
                             exp_type: str = None,
                             output_dir: str = None,
                             show: bool = True):
    """
    Produce two sets of 5 confusion matrix figures, saved to separate subfolders:
      - confusion_markov/    : Type A — Markov vs Non-Markov class detection
      - confusion_identity/  : Type B — Markov detection + exact identity correct
    """
    df = pd.read_csv(csv_file)

    if exp_type is None:
        exp_type = 'overall'

    df_f = df[df['type'] == exp_type].copy()

    if models:
        df_f = df_f[df_f['model'].isin(models)]

    all_models = list(df_f['model'].unique())
    all_rounds = sorted(df_f['rounds'].unique())

    required = {'markov_tp', 'markov_fp', 'markov_fn', 'markov_tn',
                 'markov_tp_strict', 'markov_fp_strict', 'markov_fn_strict', 'markov_tn_strict'}
    if not required.issubset(set(df_f.columns)):
        print("警告: CSV 缺少欄位，請先重跑 evaluate + export。")
        return

    base_dir = output_dir or os.path.join(os.path.dirname(__file__), '..', 'plots')

    # Type A: Markov class detection only
    dir_a = os.path.join(base_dir, 'confusion_markov')
    print(f"\n── Type A: Markov class detection → {dir_a}")
    _save_confusion_set(
        df_f, all_models, all_rounds, exp_type,
        'markov_tp', 'markov_fp', 'markov_fn', 'markov_tn',
        'Markov Detection (class only)',
        dir_a, show
    )

    # Type B: strict — correct Markov identity required for TP
    dir_b = os.path.join(base_dir, 'confusion_identity')
    print(f"\n── Type B: Markov detection + exact identity → {dir_b}")
    _save_confusion_set(
        df_f, all_models, all_rounds, exp_type,
        'markov_tp_strict', 'markov_fp_strict', 'markov_fn_strict', 'markov_tn_strict',
        'Markov Detection (exact identity required for TP)',
        dir_b, show
    )


def main():
    parser = argparse.ArgumentParser(
        description='绘制评估指标随trajectory长度变化的趋势图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制所有指标的单独图表
  python tools/plot_metrics.py
  
  # 绘制特定指标
  python tools/plot_metrics.py --metrics acc mda union
  
  # 只绘制特定模型
  python tools/plot_metrics.py --models gpt-5-mini deepseek-chat
  
  # 绘制type1的指标
  python tools/plot_metrics.py --type type1
  
  # 绘制组合图（所有指标在一张图上）
  python tools/plot_metrics.py --combined
  
  # 自定义输出目录
  python tools/plot_metrics.py --output my_plots/
        """
    )
    
    parser.add_argument('--metrics', nargs='+',
                       help='要绘制的指标列表（默认: 所有指标；可用 all 表示全部）')
    parser.add_argument('--models', nargs='+',
                       help='要包含的模型列表（默认: 所有模型）')
    parser.add_argument('--type', type=str,
                       choices=['overall', 'type1', 'type2'],
                       default='overall',
                       help='实验类型（默认: overall）')
    parser.add_argument('--output', type=str,
                       help='输出目录（默认: plots/）')
    parser.add_argument('--combined', action='store_true',
                       help='绘制组合图（所有指标在一张图）')
    parser.add_argument('--confusion', action='store_true',
                       help='绘制 Markov 混淆矩陣圖（TP/FP/FN/TN heatmap）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图表，仅保存文件')
    
    args = parser.parse_args()
    
    # 自动查找CSV文件
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'metrics_export.csv')
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    print("\n" + "="*80)
    print("评估指标趋势图绘制工具")
    print("="*80)
    print(f"数据源: {csv_path}")
    print(f"实验类型: {args.type}")
    if args.models:
        print(f"模型: {', '.join(args.models)}")
    if args.metrics:
        print(f"指标: {', '.join(args.metrics)}")
    print("="*80 + "\n")
    
    show = not args.no_show

    if args.confusion:
        plot_confusion_matrices(
            csv_path,
            models=args.models,
            exp_type=args.type,
            output_dir=args.output,
            show=show
        )
    elif args.combined:
        # 绘制组合图
        plot_multiple_metrics(
            csv_path,
            metrics=args.metrics,
            models=args.models,
            exp_type=args.type,
            output_dir=args.output,
            show=show
        )
    else:
        # 绘制单独图表
        plot_metrics_by_rounds(
            csv_path,
            metrics=args.metrics,
            models=args.models,
            exp_type=args.type,
            output_dir=args.output,
            show=show
        )
    
    print("\n✓ 绘制完成！")


if __name__ == "__main__":
    main()
