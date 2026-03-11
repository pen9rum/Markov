"""
绘制评估指标随trajectory长度变化的趋势图
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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
    if metrics is None:
        metrics = ['acc', 'mda', 'tv', 'wr_gap', 'ce', 'brier', 'evloss', 'union']
    
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
    
    # 指标的中文名称和描述
    metric_info = {
        'acc': ('准确率 (ACC)', 'Accuracy', 'higher is better'),
        'mda': ('方向准确率 (MDA)', 'Move Direction Accuracy', 'higher is better'),
        'tv': ('总变差 (TV)', 'Total Variation Distance', 'lower is better'),
        'wr_gap': ('胜率差距 (WR_GAP)', 'Win Rate Gap', 'lower is better'),
        'ce': ('交叉熵 (CE)', 'Cross Entropy', 'lower is better'),
        'brier': ('Brier分数', 'Brier Score', 'lower is better'),
        'evloss': ('期望值损失 (EVLoss)', 'Expected Value Loss', 'lower is better'),
        'union': ('联合损失 (Union)', 'Union Loss', 'lower is better')
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
        
        # 设置标题和标签
        title_cn, title_en, direction = metric_info.get(metric, (metric.upper(), metric, ''))
        plt.title(f'{title_cn} 随轮次变化趋势\n({title_en} vs Rounds - {direction})', 
                 fontsize=14, pad=15)
        plt.xlabel('游戏轮次 (Rounds)', fontsize=12)
        plt.ylabel(f'{title_cn}', fontsize=12)
        
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
    if metrics is None:
        metrics = ['acc', 'mda', 'tv', 'wr_gap', 'ce', 'brier', 'evloss', 'union']
    
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
        elif n_metrics <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
    else:
        rows, cols = layout
    
    # 指标信息
    metric_info = {
        'acc': ('ACC', 'higher ↑'),
        'mda': ('MDA', 'higher ↑'),
        'tv': ('TV', 'lower ↓'),
        'wr_gap': ('WR_GAP', 'lower ↓'),
        'ce': ('CE', 'lower ↓'),
        'brier': ('Brier', 'lower ↓'),
        'evloss': ('EVLoss', 'lower ↓'),
        'union': ('Union', 'lower ↓')
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
        
        # 设置标题和标签
        title, direction = metric_info.get(metric, (metric.upper(), ''))
        ax.set_title(f'{title} ({direction})', fontsize=11, pad=8)
        ax.set_xlabel('Rounds', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(all_rounds)
        
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    # 添加总标题
    fig.suptitle(f'评估指标随轮次变化趋势 - {exp_type}', fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    # 保存
    filename = f"all_metrics_vs_rounds_{exp_type}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ 保存组合图表: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()


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
                       help='要绘制的指标列表（默认: 所有指标）')
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
    
    if args.combined:
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
