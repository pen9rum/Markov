"""
批量解析工具
批量解析batch_results目录下的所有分析文本文件
"""
import os
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径以便导入parse_analysis
sys.path.insert(0, os.path.dirname(__file__))

from parse_analysis import parse_analysis_result


def find_analysis_files(batch_results_dir: str):
    """
    查找所有分析文本文件
    
    Args:
        batch_results_dir: batch_results目录路径
    
    Returns:
        按模型和类型组织的文件列表
    """
    files_by_model = {}
    
    batch_path = Path(batch_results_dir)
    if not batch_path.exists():
        print(f"警告: 目录不存在 {batch_results_dir}")
        return files_by_model
    
    # 遍历所有模型目录
    for model_dir in batch_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        files_by_model[model_name] = {
            'type1_non_markov': [],
            'type2_with_markov': []
        }
        
        # 遍历rounds目录（新增的层级）
        for rounds_dir in model_dir.iterdir():
            if not rounds_dir.is_dir():
                continue
            
            # 遍历type1和type2目录
            for type_dir in rounds_dir.iterdir():
                if not type_dir.is_dir():
                    continue
                
                type_name = type_dir.name
                if type_name not in ['type1_non_markov', 'type2_with_markov']:
                    continue
                
                # 查找所有.txt文件（排除汇总文件）
                for txt_file in type_dir.glob('*.txt'):
                    if not txt_file.name.startswith('_'):
                        files_by_model[model_name][type_name].append(str(txt_file))
    
    return files_by_model


def parse_batch_files(batch_results_dir: str, output_dir: str = None, 
                     include_full_text: bool = False, specific_model: str = None,
                     specific_rounds: int = None):
    """
    批量解析所有分析文件
    
    Args:
        batch_results_dir: batch_results目录路径
        output_dir: 输出目录（默认为batch_results下的parsed_output）
        include_full_text: 是否包含完整原始文本
        specific_model: 只处理特定模型（模型名称）
        specific_rounds: 只处理特定 rounds
    
    Returns:
        解析统计信息
    """
    # 查找所有文件
    files_by_model = find_analysis_files(batch_results_dir)
    
    if not files_by_model:
        print("未找到任何分析文件")
        return None
    
    # 如果指定了特定模型，只保留该模型
    if specific_model:
        if specific_model in files_by_model:
            files_by_model = {specific_model: files_by_model[specific_model]}
        else:
            print(f"错误: 未找到模型 '{specific_model}'")
            print(f"可用的模型: {', '.join(files_by_model.keys())}")
            return None
    
    # 确定输出目录
    if output_dir is None:
        # 將 parsed_output 放在 batch_results 外面（同級目錄）
        batch_parent = os.path.dirname(batch_results_dir)
        output_dir = os.path.join(batch_parent, 'parsed_output')
    
    stats = {
        'total_files': 0,
        'successful': 0,
        'failed': 0,
        'by_model': {}
    }
    
    # 处理每个模型
    for model_name, type_files in files_by_model.items():
        print(f"\n{'='*80}")
        print(f"处理模型: {model_name}")
        print(f"{'='*80}")
        
        # 如果指定了 specific_rounds，先过滤文件列表
        filtered_type1 = type_files['type1_non_markov']
        filtered_type2 = type_files['type2_with_markov']
        
        if specific_rounds is not None:
            # 过滤 type1 文件
            filtered_type1 = [
                f for f in type_files['type1_non_markov']
                if Path(f).parts[-3] == str(specific_rounds)
            ]
            # 过滤 type2 文件
            filtered_type2 = [
                f for f in type_files['type2_with_markov']
                if Path(f).parts[-3] == str(specific_rounds)
            ]
        
        model_stats = {
            'type1_total': len(filtered_type1),
            'type1_success': 0,
            'type1_failed': 0,
            'type2_total': len(filtered_type2),
            'type2_success': 0,
            'type2_failed': 0
        }

        # 解析前先清理目标输出目录，避免旧JSON残留导致结果“叠加”
        rounds_to_clean = set()
        for filepath in filtered_type1 + filtered_type2:
            path_parts = Path(filepath).parts
            rounds_to_clean.add(path_parts[-3])

        for rounds in rounds_to_clean:
            type1_output_dir = os.path.join(output_dir, model_name, rounds, 'type1_non_markov')
            type2_output_dir = os.path.join(output_dir, model_name, rounds, 'type2_with_markov')

            if os.path.exists(type1_output_dir):
                shutil.rmtree(type1_output_dir)
            if os.path.exists(type2_output_dir):
                shutil.rmtree(type2_output_dir)
        
        # 处理type1
        if filtered_type1:
            print(f"\n处理类型1（非Markov）: {len(filtered_type1)} 个文件")
            
            for filepath in filtered_type1:
                try:
                    # 从路径中提取 rounds 信息
                    # 路径格式: .../batch_results/{model}/{rounds}/type1_non_markov/xxx.txt
                    path_parts = Path(filepath).parts
                    rounds = path_parts[-3]  # rounds 在倒数第3个位置
                    
                    # 创建输出目录（保留 rounds 层级）
                    type1_output_dir = os.path.join(output_dir, model_name, rounds, 'type1_non_markov')
                    os.makedirs(type1_output_dir, exist_ok=True)
                    
                    # 读取文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 解析
                    result = parse_analysis_result(content, include_full_text=include_full_text)
                    
                    # 只有解析成功才保存 JSON
                    if result.get('parse_success'):
                        filename = Path(filepath).stem + '.json'
                        json_path = os.path.join(type1_output_dir, filename)
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        model_stats['type1_success'] += 1
                        stats['successful'] += 1
                        print(f"  ✓ {Path(filepath).name}")
                    else:
                        model_stats['type1_failed'] += 1
                        stats['failed'] += 1
                        print(f"  ✗ {Path(filepath).name}: {result.get('error', 'Unknown')}")
                    
                    stats['total_files'] += 1
                
                except Exception as e:
                    model_stats['type1_failed'] += 1
                    stats['failed'] += 1
                    stats['total_files'] += 1
                    print(f"  ✗ {Path(filepath).name}: Exception - {str(e)}")
        
        # 处理type2
        if filtered_type2:
            print(f"\n处理类型2（含Markov）: {len(filtered_type2)} 个文件")
            
            for filepath in filtered_type2:
                try:
                    # 从路径中提取 rounds 信息
                    # 路径格式: .../batch_results/{model}/{rounds}/type2_with_markov/xxx.txt
                    path_parts = Path(filepath).parts
                    rounds = path_parts[-3]  # rounds 在倒数第3个位置
                    
                    # 创建输出目录（保留 rounds 层级）
                    type2_output_dir = os.path.join(output_dir, model_name, rounds, 'type2_with_markov')
                    os.makedirs(type2_output_dir, exist_ok=True)
                    
                    # 读取文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 解析
                    result = parse_analysis_result(content, include_full_text=include_full_text)
                    
                    # 只有解析成功才保存 JSON
                    if result.get('parse_success'):
                        filename = Path(filepath).stem + '.json'
                        json_path = os.path.join(type2_output_dir, filename)
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        model_stats['type2_success'] += 1
                        stats['successful'] += 1
                        print(f"  ✓ {Path(filepath).name}")
                    else:
                        model_stats['type2_failed'] += 1
                        stats['failed'] += 1
                        print(f"  ✗ {Path(filepath).name}: {result.get('error', 'Unknown')}")
                    
                    stats['total_files'] += 1
                
                except Exception as e:
                    model_stats['type2_failed'] += 1
                    stats['failed'] += 1
                    stats['total_files'] += 1
                    print(f"  ✗ {Path(filepath).name}: Exception - {str(e)}")
        
        stats['by_model'][model_name] = model_stats
    
    return stats


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量解析batch_results目录下的所有分析文本文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 解析默认目录 (../batch_results)
  python batch_parser.py
  
  # 指定输入和输出目录
  python batch_parser.py --input ../batch_results --output ../parsed_batch
  
  # 包含完整原始文本
  python batch_parser.py --full-text
  
  # 只解析特定模型
  python batch_parser.py --model gemini-3-flash-preview
  
  # 只解析特定模型的特定 rounds
  python batch_parser.py --model deepseek-chat --rounds 200
        """
    )
    
    parser.add_argument('--input', 
                       default=os.path.join(os.path.dirname(__file__), '..', 'batch_results'),
                       help='batch_results目录路径（默认: ../batch_results）')
    parser.add_argument('--output',
                       help='输出目录（默认: batch_results/parsed_output）')
    parser.add_argument('--full-text', action='store_true',
                       help='在JSON中包含完整的原始分析文本')
    parser.add_argument('--model',
                       help='只解析指定模型的文件')
    parser.add_argument('--rounds', type=int,
                       help='只解析指定 rounds 的文件')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("批量解析工具")
    print(f"{'='*80}")
    print(f"输入目录: {args.input}")
    
    if args.model:
        print(f"只处理模型: {args.model}")
    
    if args.rounds:
        print(f"只处理 rounds: {args.rounds}")
    
    # 执行批量解析
    stats = parse_batch_files(args.input, args.output, args.full_text, args.model, args.rounds)
    
    if stats:
        # 打印统计信息
        print(f"\n{'='*80}")
        print("解析完成")
        print(f"{'='*80}")
        print(f"总文件数: {stats['total_files']}")
        print(f"成功: {stats['successful']}")
        print(f"失败: {stats['failed']}")
        if stats['total_files'] > 0:
            print(f"成功率: {stats['successful']/stats['total_files']*100:.1f}%")
        else:
            print(f"成功率: N/A (未找到任何文件)")
        
        print(f"\n按模型统计:")
        for model_name, model_stats in stats['by_model'].items():
            print(f"\n  {model_name}:")
            print(f"    类型1（非Markov）: {model_stats['type1_success']}/{model_stats['type1_total']} 成功")
            print(f"    类型2（含Markov）: {model_stats['type2_success']}/{model_stats['type2_total']} 成功")
        
        if args.output:
            print(f"\n解析结果已保存到: {args.output}")
        else:
            print(f"\n解析结果已保存到: {os.path.join(args.input, 'parsed_output')}")


if __name__ == "__main__":
    main()