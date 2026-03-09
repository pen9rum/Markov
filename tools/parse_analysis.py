import re
import json
import sys
import os
from typing import Dict, Any, Optional


def parse_final_answer(text: str) -> Optional[Dict[str, Any]]:
    """
    从LLM输出中解析最终答案部分

    支持格式:
    1. 概率格式: Player1: G, 0.5, 0.3, 0.2
    2. 次數格式: Player1: G, Rock 0, Paper 50, Scissors 50
    3. 混合格式: Player1: G, 0, 50, 50
    4. count格式: Player1: G, Rock count 0, Paper count 50, Scissors count 50
    5. 等号格式: Player1: G, Rock count = 0, Paper count = 50, Scissors count = 50
    6. 百分比分布格式:
       Player1: P (...), empirical distribution 15% Rock / 27% Paper / 58% Scissors
    7. Identity格式:
       Player1: Identity D, Rock count 33, Paper count 33, Scissors count 34

    解析策略:
    1. 优先找 "Identities and counts:"
    2. 再找 "Counts ..."
    3. 再找 "Identities:"
    4. 再找 "Final Answer:"
    5. 如果都失败，就直接从全文抓最后出现的 Player1 / Player2 行
    """

    def try_parse_players(block_text: str) -> Dict[str, Any]:
        players_data = {}

        player_pattern_labeled = (
            r'Player\s*([12])\s*:\s*'
            r'(?:Identity\s+)?'
            r'([A-Z])'
            r'(?:\s*\([^)]*\))?'
            r'(?:\s*[.,:-])?\s*'
            r'Rock\s*(?:count\s*)?(?:=\s*)?([\d.]+),?\s*'
            r'Paper\s*(?:count\s*)?(?:=\s*)?([\d.]+),?\s*'
            r'Scissors\s*(?:count\s*)?(?:=\s*)?([\d.]+)'
        )

        player_pattern_simple = (
            r'Player\s*([12])\s*:\s*'
            r'(?:Identity\s+)?'
            r'([A-Z])'
            r'(?:\s*\([^)]*\))?'
            r'(?:\s*[.,:-])?\s*'
            r'([\d.]+),?\s*([\d.]+),?\s*([\d.]+)'
        )

        player_pattern_percent = (
            r'Player\s*([12])\s*:\s*'
            r'(?:Identity\s+)?'
            r'([A-Z])'
            r'(?:\s*\([^)]*\))?'
            r'.*?([\d.]+)%\s*Rock\s*/\s*([\d.]+)%\s*Paper\s*/\s*([\d.]+)%\s*Scissors'
        )

        valid_identities = set("ABCDEFGHIJKLMNOPXYZ")

        def build_player_data(identity, rock_val, paper_val, scissors_val, is_percent=False):
            if is_percent:
                return {
                    "identity": identity,
                    "counts": {
                        "rock": int(round(rock_val)),
                        "paper": int(round(paper_val)),
                        "scissors": int(round(scissors_val))
                    },
                    "probabilities": {
                        "rock": rock_val / 100.0,
                        "paper": paper_val / 100.0,
                        "scissors": scissors_val / 100.0
                    }
                }

            total = rock_val + paper_val + scissors_val

            if total > 3:
                rock_prob = rock_val / total if total > 0 else 0
                paper_prob = paper_val / total if total > 0 else 0
                scissors_prob = scissors_val / total if total > 0 else 0

                counts = {
                    "rock": int(round(rock_val)),
                    "paper": int(round(paper_val)),
                    "scissors": int(round(scissors_val))
                }
            else:
                rock_prob = rock_val
                paper_prob = paper_val
                scissors_prob = scissors_val

                counts = {
                    "rock": int(round(rock_val * 100)),
                    "paper": int(round(paper_val * 100)),
                    "scissors": int(round(scissors_val * 100))
                }

            return {
                "identity": identity,
                "counts": counts,
                "probabilities": {
                    "rock": rock_prob,
                    "paper": paper_prob,
                    "scissors": scissors_prob
                }
            }

        for match in re.finditer(player_pattern_labeled, block_text, re.IGNORECASE):
            player_num = match.group(1)
            identity = match.group(2).upper()
            rock_val = float(match.group(3))
            paper_val = float(match.group(4))
            scissors_val = float(match.group(5))

            if identity not in valid_identities:
                continue

            players_data[f"player{player_num}"] = build_player_data(
                identity, rock_val, paper_val, scissors_val
            )

        if len(players_data) < 2:
            for match in re.finditer(player_pattern_simple, block_text, re.IGNORECASE):
                player_num = match.group(1)
                identity = match.group(2).upper()
                rock_val = float(match.group(3))
                paper_val = float(match.group(4))
                scissors_val = float(match.group(5))

                if identity not in valid_identities:
                    continue

                key = f"player{player_num}"
                if key not in players_data:
                    players_data[key] = build_player_data(
                        identity, rock_val, paper_val, scissors_val
                    )

        if len(players_data) < 2:
            for match in re.finditer(player_pattern_percent, block_text, re.IGNORECASE | re.DOTALL):
                player_num = match.group(1)
                identity = match.group(2).upper()
                rock_pct = float(match.group(3))
                paper_pct = float(match.group(4))
                scissors_pct = float(match.group(5))

                if identity not in valid_identities:
                    continue

                key = f"player{player_num}"
                if key not in players_data:
                    players_data[key] = build_player_data(
                        identity, rock_pct, paper_pct, scissors_pct, is_percent=True
                    )

        return players_data

    identities_counts_match = re.search(
        r'Identities and counts:\s*\n(.*?)(?:\n\s*\n|={3,}|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if identities_counts_match:
        block = identities_counts_match.group(1).strip()
        players_data = try_parse_players(block)
        if len(players_data) == 2:
            return players_data

    counts_match = re.search(
        r'Counts.*?:\s*\n(.*?)(?:\n\s*\n|Justification summary:|Identities:|Brief justification:|Predicted|={3,}|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if counts_match:
        block = counts_match.group(1).strip()
        players_data = try_parse_players(block)
        if len(players_data) == 2:
            return players_data

    identities_match = re.search(
        r'Identities:\s*\n(.*?)(?:\n\s*\n|Predicted move probabilities|Predicted next-move probabilities|={3,}|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if identities_match:
        block = identities_match.group(1).strip()
        players_data = try_parse_players(block)
        if len(players_data) == 2:
            return players_data

    final_answer_match = re.search(
        r'(?:\*\*Final Answer\*\*:|\*\*Final Answer:\*\*|Final Answer:)\s*\n(.*?)(?:={3,}|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if final_answer_match:
        block = final_answer_match.group(1).strip()
        players_data = try_parse_players(block)
        if len(players_data) == 2:
            return players_data

    player_line_pattern = r'Player\s*([12])\s*:\s*.*'
    all_player_lines = list(re.finditer(player_line_pattern, text, re.IGNORECASE))

    if all_player_lines:
        last_p1 = None
        last_p2 = None

        for m in reversed(all_player_lines):
            line = m.group(0).strip()
            num_match = re.search(r'Player\s*([12])\s*:', line, re.IGNORECASE)
            if not num_match:
                continue

            player_num = num_match.group(1)
            if player_num == "1" and last_p1 is None:
                last_p1 = line
            elif player_num == "2" and last_p2 is None:
                last_p2 = line

            if last_p1 and last_p2:
                break

        if last_p1 and last_p2:
            fallback_block = f"{last_p1}\n{last_p2}"
            players_data = try_parse_players(fallback_block)
            if len(players_data) == 2:
                return players_data

    print("错误: 无法解析 Player1 / Player2 的最终答案")
    return None


def detect_markov_player(text: str) -> Optional[str]:
    """
    从分析文本中检测是否有马可夫玩家

    Returns:
        "player1", "player2", "neither", 或 None
    """
    lower_text = text.lower()

    neither_patterns = [
        r'no markov',
        r'neither.*markov',
        r'not markov',
        r'no player.*markov',
        r'both.*not markov',
        r'player\s*1.*not markov.*player\s*2.*not markov',
        r'player\s*2.*not markov.*player\s*1.*not markov',
        r'no markov player detected',
        r'no markov player',
    ]
    for pattern in neither_patterns:
        if re.search(pattern, lower_text):
            return "neither"

    player1_positive = [
        r'player\s*1\s+is\s+(a\s+)?markov',
        r'player\s*1\s+appears\s+to\s+be\s+(a\s+)?markov',
        r'player\s*1\s*=\s*[xyz]',
        r'markov player.*player\s*1',
        r'player1\s*=\s*[xyz]',
    ]
    player2_positive = [
        r'player\s*2\s+is\s+(a\s+)?markov',
        r'player\s*2\s+appears\s+to\s+be\s+(a\s+)?markov',
        r'player\s*2\s*=\s*[xyz]',
        r'markov player.*player\s*2',
        r'player2\s*=\s*[xyz]',
    ]

    for pattern in player1_positive:
        m = re.search(pattern, lower_text)
        if m:
            snippet = lower_text[max(0, m.start() - 30): m.end() + 30]
            if "not markov" not in snippet:
                return "player1"

    for pattern in player2_positive:
        m = re.search(pattern, lower_text)
        if m:
            snippet = lower_text[max(0, m.start() - 30): m.end() + 30]
            if "not markov" not in snippet:
                return "player2"

    return None


def parse_ground_truth(text: str) -> Optional[Dict[str, Any]]:
    """
    從分析文件中解析真實的玩家身份和實際分布

    Returns:
        包含真實數據的字典，如果解析失败則返回None
    """
    ground_truth = {}

    match_pattern = r'Match:\s*([A-Z])\s+vs\s+([A-Z])'
    match_result = re.search(match_pattern, text, re.IGNORECASE)

    if match_result:
        ground_truth['player1_identity'] = match_result.group(1).upper()
        ground_truth['player2_identity'] = match_result.group(2).upper()

    player1_dist_pattern = (
        r'Player1 Actual Distribution:\s*\n'
        r'\s*Rock:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n'
        r'\s*Paper:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n'
        r'\s*Scissors:\s*(\d+)\s*\(([0-9.]+)%\)'
    )
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

    player2_dist_pattern = (
        r'Player2 Actual Distribution:\s*\n'
        r'\s*Rock:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n'
        r'\s*Paper:\s*(\d+)\s*\(([0-9.]+)%\)\s*\n'
        r'\s*Scissors:\s*(\d+)\s*\(([0-9.]+)%\)'
    )
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

    ground_truth = parse_ground_truth(analysis_text)
    if ground_truth:
        result["ground_truth"] = ground_truth

    players_data = parse_final_answer(analysis_text)
    if players_data:
        result["parse_success"] = True
        result["predictions"] = players_data
    else:
        result["error"] = "Failed to parse Final Answer section"

    markov_player = detect_markov_player(analysis_text)
    if markov_player:
        result["markov_detection"] = markov_player

    if include_full_text:
        result["full_analysis"] = analysis_text

    return result


def parse_from_file(
    filepath: str,
    output_json: bool = True,
    output_file: Optional[str] = None,
    include_full_text: bool = False
) -> Dict[str, Any]:
    """
    从文件中读取并解析分析结果
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    result = parse_analysis_result(content, include_full_text=include_full_text)

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
    import argparse

    parser = argparse.ArgumentParser(
        description='解析LLM分析结果并转换为结构化JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python parse_analysis.py input.txt
  python parse_analysis.py input.txt -o output.json
  python parse_analysis.py input.txt -o output.json --full-text
  cat analysis.txt | python parse_analysis.py -
        """
    )

    parser.add_argument(
        'input',
        help='输入文件路径（使用 - 表示从stdin读取）'
    )
    parser.add_argument(
        '-o', '--output',
        help='输出JSON文件路径（默认输出到stdout）'
    )
    parser.add_argument(
        '--full-text',
        action='store_true',
        help='在JSON中包含完整的原始分析文本'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='只显示解析状态，不输出JSON'
    )

    args = parser.parse_args()

    if args.input == '-':
        content = sys.stdin.read()
    else:
        if not os.path.exists(args.input):
            print(f"错误: 文件不存在: {args.input}", file=sys.stderr)
            sys.exit(1)

        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()

    result = parse_analysis_result(content, include_full_text=args.full_text)

    if not args.no_json:
        json_str = json.dumps(result, ensure_ascii=False, indent=2)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✓ 结果已保存到: {args.output}", file=sys.stderr)
        else:
            print(json_str)

    if result['parse_success']:
        print("✓ 解析成功", file=sys.stderr)
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