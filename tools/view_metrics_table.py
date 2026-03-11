"""
CSV 指標表格顯示工具
以易讀的表格格式顯示評估指標
"""
import csv
import argparse
from pathlib import Path


def format_number(value, precision=4):
    """格式化數字"""
    if value is None or value == '':
        return '-'
    try:
        num = float(value)
        if precision == 0:
            return f"{int(num)}"
        return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def print_table(rows, headers, col_widths=None):
    """打印表格"""
    if not rows:
        print("無數據")
        return
    
    # 自動計算列寬
    if col_widths is None:
        col_widths = {}
        for i, header in enumerate(headers):
            col_widths[i] = len(header)
            for row in rows:
                if i < len(row):
                    col_widths[i] = max(col_widths[i], len(str(row[i])))
    
    # 打印表頭
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # 打印數據行
    for row in rows:
        row_line = " | ".join(str(row[i]).ljust(col_widths[i]) if i < len(row) else " " * col_widths[i] 
                              for i in range(len(headers)))
        print(row_line)


def view_metrics(csv_file, filter_type=None, filter_model=None, filter_rounds=None, 
                 metrics=None, sort_by=None, reverse=False):
    """
    查看指標表格
    
    Args:
        csv_file: CSV 文件路徑
        filter_type: 過濾類型 (overall/type1_non_markov/type2_with_markov)
        filter_model: 過濾模型名稱
        filter_rounds: 過濾 rounds
        metrics: 要顯示的指標列表
        sort_by: 排序依據的指標
        reverse: 是否降序排列
    """
    # 讀取 CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if not data:
        print("CSV 文件為空")
        return
    
    # 應用過濾器
    filtered_data = data
    
    if filter_type:
        filtered_data = [row for row in filtered_data if row['type'] == filter_type]
    
    if filter_model:
        filtered_data = [row for row in filtered_data if row['model'] == filter_model]
    
    if filter_rounds:
        filtered_data = [row for row in filtered_data if row['rounds'] == str(filter_rounds)]
    
    if not filtered_data:
        print("沒有符合條件的數據")
        return
    
    # 確定要顯示的列
    all_columns = list(data[0].keys())
    
    if metrics:
        # 總是包含 model, rounds, type
        base_cols = ['model', 'rounds', 'type', 'samples']
        display_cols = base_cols + [m for m in metrics if m not in base_cols]
        # 只保留存在的列
        display_cols = [col for col in display_cols if col in all_columns]
    else:
        display_cols = all_columns
    
    # 排序
    if sort_by and sort_by in display_cols:
        try:
            filtered_data.sort(key=lambda x: float(x.get(sort_by, 0) or 0), reverse=reverse)
        except ValueError:
            filtered_data.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
    
    # 準備表格數據
    headers = display_cols
    rows = []
    
    for row_data in filtered_data:
        row = []
        for col in display_cols:
            value = row_data.get(col, '')
            # 格式化數字列
            if col in ['acc', 'mda', 'tv', 'wr_gap', 'ce', 'brier', 'evloss', 'union']:
                row.append(format_number(value, precision=4))
            elif col in ['samples', 'skipped_count', 'rounds']:
                row.append(format_number(value, precision=0))
            else:
                row.append(value)
        rows.append(row)
    
    # 打印表格
    print_table(rows, headers)
    print(f"\n總計: {len(rows)} 筆資料")


def main():
    parser = argparse.ArgumentParser(description='CSV 指標表格顯示工具')
    parser.add_argument('csv_file', nargs='?', default='metrics_export.csv',
                       help='CSV 文件路徑 (默認: metrics_export.csv)')
    parser.add_argument('--type', choices=['overall', 'type1_non_markov', 'type2_with_markov'],
                       help='過濾類型')
    parser.add_argument('--model', help='過濾模型名稱')
    parser.add_argument('--rounds', type=int, help='過濾 rounds')
    parser.add_argument('--metrics', nargs='+',
                       help='要顯示的指標 (例: acc mda tv union)')
    parser.add_argument('--sort', help='排序依據的指標')
    parser.add_argument('--reverse', action='store_true', help='降序排列')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"錯誤: 找不到文件 {csv_path}")
        return
    
    print("=" * 80)
    print("評估指標表格")
    print("=" * 80)
    
    view_metrics(
        csv_file=csv_path,
        filter_type=args.type,
        filter_model=args.model,
        filter_rounds=args.rounds,
        metrics=args.metrics,
        sort_by=args.sort,
        reverse=args.reverse
    )


if __name__ == '__main__':
    main()
