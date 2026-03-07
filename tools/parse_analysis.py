"""
解析LLM分析结果的工具
从文本分析中提取结构化的JSON数据
"""
import re
import json
import sys
import os
from typing import Dict, Any, Optional


def parse_final_answer(text: str) -> Optional[Dict[str, Any]]:
    """
    从LLM输出中解析Final Answer部分
    
    支持格式:
    1. 概率格式: Player1: G, 0.5, 0.3, 0.2
    2. 次數格式: Player1: G, Rock 0, Paper 50, Scissors 50
    3. 混合格式: Player1: G, 0, 50, 50
    
    Args:
        text: LLM的完整输出文本
    
    Returns:
        解析后的JSON结构，如果解析失败则返回None
    """
    # 查找Final Answer部分（支持加粗的**Final Answer:**）
    final_answer_match = re.search(r'\*\*Final Answer:\*\*\s*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if not final_answer_match:
        final_answer_match = re.search(r'Final Answer:\s*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if not final_answer_match:
        print("警告: 未找到 'Final Answer:' 标记")
        return None
    
    final_answer_text = final_answer_match.group(1).strip()
    
    # 解析Player1和Player2的信息
    # 支持多種格式:
    # 格式1: Player1: G, 0.5, 0.3, 0.2
    # 格式2: Player1: G, Rock 0, Paper 50, Scissors 50
    # 格式3: Player1: G, 0, 50, 50
    # 格式4: Player1: G, Rock count 31, Paper count 30, Scissors count 39
    
    # 先嘗試格式2和4（帶標籤的次數格式，可選 "count" 關鍵字）
    player_pattern_labeled = r'Player([12]):\s*([A-Z]),?\s*Rock\s+(?:count\s+)?([\d.]+),?\s*Paper\s+(?:count\s+)?([\d.]+),?\s*Scissors\s+(?:count\s+)?([\d.]+)'
    
    # 再嘗試格式1和3（純數字格式）
    player_pattern_simple = r'Player([12]):\s*([A-Z]),?\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)'
    
    players_data = {}
    
    # 先嘗試帶標籤的格式
    for match in re.finditer(player_pattern_labeled, final_answer_text, re.IGNORECASE):
        player_num = match.group(1)
        identity = match.group(2).upper()
        rock_val = float(match.group(3))
        paper_val = float(match.group(4))
        scissors_val = float(match.group(5))
        
        # 验证identity是否有效
        valid_identities = set('ABCDEFGHIJKLMNOPXYZ')
        if identity not in valid_identities:
            print(f"警告: Player{player_num} 的身份 '{identity}' 无效")
            continue
        
        # 如果數值大於1，視為次數，需要轉換為概率
        total = rock_val + paper_val + scissors_val
        if total > 3:  # 很可能是次數而非概率
            rock_prob = rock_val / total if total > 0 else 0
            paper_prob = paper_val / total if total > 0 else 0
            scissors_prob = scissors_val / total if total > 0 else 0
        else:
            rock_prob = rock_val
            paper_prob = paper_val
            scissors_prob = scissors_val
        
        players_data[f'player{player_num}'] = {
            'identity': identity,
            'counts': {
                'rock': int(rock_val),
                'paper': int(paper_val),
                'scissors': int(scissors_val)
            },
            'probabilities': {
                'rock': rock_prob,
                'paper': paper_prob,
                'scissors': scissors_prob
            }
        }
    
    # 如果沒有找到帶標籤的格式，嘗試簡單格式
    if len(players_data) == 0:
        for match in re.finditer(player_pattern_simple, final_answer_text, re.IGNORECASE):
            player_num = match.group(1)
            identity = match.group(2).upper()
            rock_val = float(match.group(3))
            paper_val = float(match.group(4))
            scissors_val = float(match.group(5))
            
            # 验证identity是否有效
            valid_identities = set('ABCDEFGHIJKLMNOPXYZ')
            if identity not in valid_identities:
                print(f"警告: Player{player_num} 的身份 '{identity}' 无效")
                continue
            
            # 如果數值大於1，視為次數，需要轉換為概率
            total = rock_val + paper_val + scissors_val
            if total > 3:  # 很可能是次數而非概率
                rock_prob = rock_val / total if total > 0 else 0
                paper_prob = paper_val / total if total > 0 else 0
                scissors_prob = scissors_val / total if total > 0 else 0
            else:
                rock_prob = rock_val
                paper_prob = paper_val
                scissors_prob = scissors_val
            
            players_data[f'player{player_num}'] = {
                'identity': identity,
                'counts': {
                    'rock': int(rock_val),
                    'paper': int(paper_val),
                    'scissors': int(scissors_val)
                },
                'probabilities': {
                    'rock': rock_prob,
                    'paper': paper_prob,
                    'scissors': scissors_prob
                }
            }
    
    if len(players_data) != 2:
        print(f"警告: 只解析到 {len(players_data)} 个玩家的信息")
        return None
    
    return players_data


def detect_markov_player(text: str) -> Optional[str]:
    """
    从分析文本中检测是否有马可夫玩家
    
    Returns:
        "player1", "player2", "neither", 或 None (无法确定)
    """
    # 常见的标识马可夫玩家的模式
    patterns = [
        r'Player\s*1\s+is\s+(a\s+)?Markov',
        r'Player\s*2\s+is\s+(a\s+)?Markov',
        r'(No|Neither|Both).{0,20}Markov',
        r'Markov\s+player.{0,20}Player\s*([12])',
    ]
    
    markov_info = {
        'player1': False,
        'player2': False,
        'neither': False
    }
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if 'Player 1' in match.group(0) or 'Player1' in match.group(0):
                markov_info['player1'] = True
            elif 'Player 2' in match.group(0) or 'Player2' in match.group(0):
                markov_info['player2'] = True
            elif 'No' in match.group(0) or 'Neither' in match.group(0):
                markov_info['neither'] = True
    
    # 判断结果
    if markov_info['neither']:
        return "neither"
    elif markov_info['player1'] and not markov_info['player2']:
        return "player1"
    elif markov_info['player2'] and not markov_info['player1']:
        return "player2"
    
    return None


def parse_ground_truth(text: str) -> Optional[Dict[str, Any]]:
    """
    從分析文件中解析真實的玩家身份和實際分布
    
    Returns:
        包含真實數據的字典，如果解析失败則返回None
    """
    ground_truth = {}
    
    # 解析 Match 行，提取真實玩家身份
    # 格式: Match: G vs P
    match_pattern = r'Match:\s*([A-Z])\s+vs\s+([A-Z])'
    match_result = re.search(match_pattern, text, re.IGNORECASE)
    
    if match_result:
        ground_truth['player1_identity'] = match_result.group(1).upper()
        ground_truth['player2_identity'] = match_result.group(2).upper()
    
    # 解析 Player1 Actual Distribution
    # 格式:
    # Player1 Actual Distribution:
    #   Rock: 0 (0.0%)
    #   Paper: 250 (50.0%)
    #   Scissors: 250 (50.0%)
    
    player1_dist_pattern = r'Player1 Actual Distribution:\s*\n\s*Rock:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n\s*Paper:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n\s*Scissors:\s*(\d+)\s*\(([0-9.]+)%\)'
    player1_match = re.search(player1_dist_pattern, text, re.IGNORECASE)
    
    if player1_match:
        ground_truth['player1'] = {
            'counts': {
                'rock': int(player1_match.group(1)),
                'paper': int(player1_match.group(3)),
                'scissors': int(player1_match.group(5))
            },
            'probabilities': {
                'rock': float(player1_match.group(2)) / 100,
                'paper': float(player1_match.group(4)) / 100,
                'scissors': float(player1_match.group(6)) / 100
            }
        }
    
    # 解析 Player2 Actual Distribution
    player2_dist_pattern = r'Player2 Actual Distribution:\s*\n\s*Rock:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n\s*Paper:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n\s*Scissors:\s*(\d+)\s*\(([0-9.]+)%\)'
    player2_match = re.search(player2_dist_pattern, text, re.IGNORECASE)
    
    if player2_match:
        ground_truth['player2'] = {
            'counts': {
                'rock': int(player2_match.group(1)),
                'paper': int(player2_match.group(3)),
                'scissors': int(player2_match.group(5))
            },
            'probabilities': {
                'rock': float(player2_match.group(2)) / 100,
                'paper': float(player2_match.group(4)) / 100,
                'scissors': float(player2_match.group(6)) / 100
            }
        }
    
    return ground_truth if ground_truth else None


def parse_analysis_result(analysis_text: str, include_full_text: bool = False) -> Dict[str, Any]:
    """
    完整解析分析结果
    
    Args:
        analysis_text: LLM的完整输出文本
        include_full_text: 是否在JSON中包含完整的原始文本
    
    Returns:
        包含所有解析信息的JSON结构
    """
    result = {
        "parse_success": False,
        "ground_truth": None,
        "predictions": None,
        "markov_detection": None,
        "error": None
    }
    
    # 解析真實數據 (Ground Truth)
    ground_truth = parse_ground_truth(analysis_text)
    if ground_truth:
        result["ground_truth"] = ground_truth
    
    # 解析LLM預測 (Final Answer)
    players_data = parse_final_answer(analysis_text)
    
    if players_data:
        result["parse_success"] = True
        result["predictions"] = players_data
    else:
        result["error"] = "Failed to parse Final Answer section"
    
    # 检测马可夫玩家
    markov_player = detect_markov_player(analysis_text)
    if markov_player:
        result["markov_detection"] = markov_player
    
    # 可选：包含完整文本
    if include_full_text:
        result["full_analysis"] = analysis_text
    
    return result


def parse_from_file(filepath: str, output_json: bool = True, 
                   output_file: Optional[str] = None,
                   include_full_text: bool = False) -> Dict[str, Any]:
    """
    从文件中读取并解析分析结果
    
    Args:
        filepath: 输入文件路径
        output_json: 是否输出JSON格式
        output_file: 输出文件路径（如果为None则输出到stdout）
        include_full_text: 是否包含完整原始文本
    
    Returns:
        解析结果字典
    """
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析
    result = parse_analysis_result(content, include_full_text=include_full_text)
    
    # 输出
    if output_json:
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"结果已保存到: {output_file}")
        else:
            print(json_str)
    
    return result


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='解析LLM分析结果并转换为结构化JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从文件解析并输出到stdout
  python parse_analysis.py input.txt
  
  # 保存到JSON文件
  python parse_analysis.py input.txt -o output.json
  
  # 包含完整原始文本
  python parse_analysis.py input.txt -o output.json --full-text
  
  # 从stdin读取
  cat analysis.txt | python parse_analysis.py -
        """
    )
    
    parser.add_argument('input', 
                       help='输入文件路径（使用 - 表示从stdin读取）')
    parser.add_argument('-o', '--output', 
                       help='输出JSON文件路径（默认输出到stdout）')
    parser.add_argument('--full-text', action='store_true',
                       help='在JSON中包含完整的原始分析文本')
    parser.add_argument('--no-json', action='store_true',
                       help='只显示解析状态，不输出JSON')
    
    args = parser.parse_args()
    
    # 读取输入
    if args.input == '-':
        content = sys.stdin.read()
    else:
        if not os.path.exists(args.input):
            print(f"错误: 文件不存在: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()
    
    # 解析
    result = parse_analysis_result(content, include_full_text=args.full_text)
    
    # 输出
    if not args.no_json:
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✓ 结果已保存到: {args.output}", file=sys.stderr)
        else:
            print(json_str)
    
    # 输出解析状态
    if result['parse_success']:
        print(f"✓ 解析成功", file=sys.stderr)
        if result.get('ground_truth'):
            gt = result['ground_truth']
            if 'player1_identity' in gt and 'player2_identity' in gt:
                print(f"  真實: {gt['player1_identity']} vs {gt['player2_identity']}", file=sys.stderr)
        if result.get('predictions'):
            for player_key, player_data in result['predictions'].items():
                print(f"  預測 {player_key}: {player_data['identity']}", file=sys.stderr)
    else:
        print(f"✗ 解析失败: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
