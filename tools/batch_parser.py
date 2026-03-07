"""
批量解析工具
批量解析batch_results目录下的所有分析文本文件
"""
import os
import json
import sys
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
        
        # 遍历type1和type2目录
        for type_dir in model_dir.iterdir():
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
                     include_full_text: bool = False):
    """
    批量解析所有分析文件
    
    Args:
        batch_results_dir: batch_results目录路径
        output_dir: 输出目录（默认为batch_results下的parsed_output）
        include_full_text: 是否包含完整原始文本
    
    Returns:
        解析统计信息
    """
    # 查找所有文件
    files_by_model = find_analysis_files(batch_results_dir)
    
    if not files_by_model:
        print("未找到任何分析文件")
        return None
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(batch_results_dir, 'parsed_output')
    
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
        
        model_stats = {
            'type1_total': len(type_files['type1_non_markov']),
            'type1_success': 0,
            'type1_failed': 0,
            'type2_total': len(type_files['type2_with_markov']),
            'type2_success': 0,
            'type2_failed': 0
        }
        
        # 处理type1
        if type_files['type1_non_markov']:
            print(f"\n处理类型1（非Markov）: {len(type_files['type1_non_markov'])} 个文件")
            type1_output_dir = os.path.join(output_dir, model_name, 'type1_non_markov')
            os.makedirs(type1_output_dir, exist_ok=True)
            
            for filepath in type_files['type1_non_markov']:
                try:
                    # 读取文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 解析
                    result = parse_analysis_result(content, include_full_text=include_full_text)
                    
                    # 保存JSON
                    filename = Path(filepath).stem + '.json'
                    json_path = os.path.join(type1_output_dir, filename)
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    if result['parse_success']:
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
        if type_files['type2_with_markov']:
            print(f"\n处理类型2（含Markov）: {len(type_files['type2_with_markov'])} 个文件")
            type2_output_dir = os.path.join(output_dir, model_name, 'type2_with_markov')
            os.makedirs(type2_output_dir, exist_ok=True)
            
            for filepath in type_files['type2_with_markov']:
                try:
                    # 读取文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 解析
                    result = parse_analysis_result(content, include_full_text=include_full_text)
                    
                    # 保存JSON
                    filename = Path(filepath).stem + '.json'
                    json_path = os.path.join(type2_output_dir, filename)
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    if result['parse_success']:
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
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("批量解析工具")
    print(f"{'='*80}")
    print(f"输入目录: {args.input}")
    
    # 如果指定了特定模型，只处理该模型
    if args.model:
        print(f"只处理模型: {args.model}")
        batch_results_dir = os.path.join(args.input, args.model)
        if args.output:
            output_dir = os.path.join(args.output, args.model)
        else:
            output_dir = os.path.join(args.input, 'parsed_output', args.model)
    else:
        batch_results_dir = args.input
        output_dir = args.output
    
    # 执行批量解析
    stats = parse_batch_files(batch_results_dir, output_dir, args.full_text)
    
    if stats:
        # 打印统计信息
        print(f"\n{'='*80}")
        print("解析完成")
        print(f"{'='*80}")
        print(f"总文件数: {stats['total_files']}")
        print(f"成功: {stats['successful']}")
        print(f"失败: {stats['failed']}")
        print(f"成功率: {stats['successful']/stats['total_files']*100:.1f}%")
        
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
