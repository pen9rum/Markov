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
    
    期望格式:
    Final Answer:
    Player1: Identity, Rock Probability, Paper Probability, Scissors Probability
    Player2: Identity, Rock Probability, Paper Probability, Scissors Probability
    
    Args:
        text: LLM的完整输出文本
    
    Returns:
        解析后的JSON结构，如果解析失败则返回None
    """
    # 查找Final Answer部分
    final_answer_match = re.search(r'Final Answer:\s*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    
    if not final_answer_match:
        print("警告: 未找到 'Final Answer:' 标记")
        return None
    
    final_answer_text = final_answer_match.group(1).strip()
    
    # 解析Player1和Player2的信息
    # 格式: Player1: X, 0.33, 0.33, 0.34
    player_pattern = r'Player([12]):\s*([A-Z]),?\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)'
    
    players_data = {}
    
    for match in re.finditer(player_pattern, final_answer_text, re.IGNORECASE):
        player_num = match.group(1)
        identity = match.group(2).upper()
        rock_prob = float(match.group(3))
        paper_prob = float(match.group(4))
        scissors_prob = float(match.group(5))
        
        # 验证identity是否有效
        valid_identities = set('ABCDEFGHIJKLMNOPXYZ')
        if identity not in valid_identities:
            print(f"警告: Player{player_num} 的身份 '{identity}' 无效")
            continue
        
        # 验证概率总和（允许一定误差）
        prob_sum = rock_prob + paper_prob + scissors_prob
        if abs(prob_sum - 1.0) > 0.05:
            print(f"警告: Player{player_num} 的概率总和为 {prob_sum}，不等于1.0")
        
        players_data[f'player{player_num}'] = {
            'identity': identity,
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
        "players": None,
        "markov_detection": None,
        "error": None
    }
    
    # 解析Final Answer
    players_data = parse_final_answer(analysis_text)
    
    if players_data:
        result["parse_success"] = True
        result["players"] = players_data
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
        if result['players']:
            for player_key, player_data in result['players'].items():
                print(f"  {player_key}: {player_data['identity']}", file=sys.stderr)
    else:
        print(f"✗ 解析失败: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
