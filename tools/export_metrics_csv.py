"""
導出評估指標為 CSV 文件
支持多個模型和多個 rounds 的批量導出
"""
import argparse
import json
import csv
from pathlib import Path


def collect_metrics(models, rounds, parsed_output_dir="parsed_output"):
    """
    收集指定模型和 rounds 的評估指標
    
    Args:
        models: 模型名稱列表
        rounds: rounds 列表
        parsed_output_dir: parsed_output 目錄路徑
    
    Returns:
        包含所有評估結果的列表
    """
    results = []
    base_path = Path(parsed_output_dir)
    
    for model in models:
        for round_num in rounds:
            # 構建路徑
            eval_path = base_path / model / str(round_num) / "evaluation_summary.json"
            
            if not eval_path.exists():
                print(f"警告: 找不到 {eval_path}")
                continue
            
            # 讀取評估結果
            try:
                with open(eval_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 添加 overall 結果
                if "overall" in data:
                    overall = data["overall"]
                    results.append({
                        "model": model,
                        "rounds": round_num,
                        "type": "overall",
                        "samples": overall.get("samples", 0),
                        "skipped_count": data.get("skipped_count", 0),
                        "acc": overall.get("ACC"),
                        "mda": overall.get("MDA"),
                        "tv": overall.get("TV"),
                        "wr_gap": overall.get("WR_gap"),
                        "ce": overall.get("CE"),
                        "brier": overall.get("Brier"),
                        "evloss": overall.get("EVLoss"),
                        "union": overall.get("Union"),
                    })
                
                # 添加 non_markov 結果
                if "non_markov" in data:
                    non_markov = data["non_markov"]
                    results.append({
                        "model": model,
                        "rounds": round_num,
                        "type": "type1_non_markov",
                        "samples": non_markov.get("samples", 0),
                        "skipped_count": data.get("skipped_count", 0),
                        "acc": non_markov.get("ACC"),
                        "mda": non_markov.get("MDA"),
                        "tv": non_markov.get("TV"),
                        "wr_gap": non_markov.get("WR_gap"),
                        "ce": non_markov.get("CE"),
                        "brier": non_markov.get("Brier"),
                        "evloss": non_markov.get("EVLoss"),
                        "union": non_markov.get("Union"),
                    })
                
                # 添加 with_markov 結果
                if "with_markov" in data:
                    with_markov = data["with_markov"]
                    results.append({
                        "model": model,
                        "rounds": round_num,
                        "type": "type2_with_markov",
                        "samples": with_markov.get("samples", 0),
                        "skipped_count": data.get("skipped_count", 0),
                        "acc": with_markov.get("ACC"),
                        "mda": with_markov.get("MDA"),
                        "tv": with_markov.get("TV"),
                        "wr_gap": with_markov.get("WR_gap"),
                        "ce": with_markov.get("CE"),
                        "brier": with_markov.get("Brier"),
                        "evloss": with_markov.get("EVLoss"),
                        "union": with_markov.get("Union"),
                    })
                
                print(f"✓ 收集完成: {model}/{round_num}")
                
            except Exception as e:
                print(f"錯誤: 讀取 {eval_path} 失敗 - {e}")
    
    return results


def export_to_csv(results, output_file):
    """
    將結果導出為 CSV 文件
    
    Args:
        results: 評估結果列表
        output_file: 輸出文件路徑
    """
    if not results:
        print("沒有數據可以導出")
        return
    
    # CSV 表頭
    fieldnames = [
        "model",
        "rounds",
        "type",
        "samples",
        "skipped_count",
        "acc",
        "mda",
        "tv",
        "wr_gap",
        "ce",
        "brier",
        "evloss",
        "union",
    ]
    
    # 寫入 CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ CSV 文件已保存: {output_file}")
    print(f"  總計 {len(results)} 行數據")


def main():
    parser = argparse.ArgumentParser(
        description='導出多個模型和 rounds 的評估指標為 CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 導出單個模型的多個 rounds
  python export_metrics_csv.py --models gpt-5-mini --rounds 100 200 500 1000
  
  # 導出多個模型的多個 rounds
  python export_metrics_csv.py --models gpt-5-mini deepseek-chat --rounds 100 200 500 1000
  
  # 指定輸出文件名
  python export_metrics_csv.py --models gpt-5-mini deepseek-chat --rounds 200 500 --output my_results.csv
        """
    )
    
    parser.add_argument('--models', nargs='+', required=True,
                       help='模型名稱列表（空格分隔）')
    parser.add_argument('--rounds', nargs='+', type=int, required=True,
                       help='rounds 列表（空格分隔）')
    parser.add_argument('--output', type=str, default='metrics_export.csv',
                       help='輸出 CSV 文件名（默認: metrics_export.csv）')
    parser.add_argument('--input', type=str, default='parsed_output',
                       help='parsed_output 目錄路徑（默認: parsed_output）')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("評估指標 CSV 導出工具")
    print(f"{'='*80}")
    print(f"模型: {', '.join(args.models)}")
    print(f"Rounds: {', '.join(map(str, args.rounds))}")
    print(f"輸出文件: {args.output}")
    print(f"{'='*80}\n")
    
    # 收集數據
    results = collect_metrics(args.models, args.rounds, args.input)
    
    # 導出 CSV
    if results:
        export_to_csv(results, args.output)
    else:
        print("\n未找到任何數據，請檢查模型名稱和 rounds 是否正確")


if __name__ == "__main__":
    main()
