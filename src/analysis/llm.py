"""
LLM模块 - 用于分析玩家行为 (使用Qwen API)

支持的模型：
- qwen-turbo: 快速响应，性价比高
- qwen-plus: 平衡性能（推荐）
- qwen-max: 最强性能
- qwen-max-latest: 最新版本（可能包括3.5）
- qwen2.5-72b-instruct: Qwen 2.5 开源版本
- qwen2.5-32b-instruct: Qwen 2.5 中等规模
- qwen2.5-14b-instruct: Qwen 2.5 小规模
- qwen2.5-7b-instruct: Qwen 2.5 最小规模
"""
from typing import Tuple, Dict, Any
import os
import requests
import json


def get_response(prompt: str,
                 model_name: str = "qwen-plus",
                 api_key: str = None,
                 api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 **kwargs) -> Tuple[dict, str]:
    """
    调用Qwen API获取响应
    
    Args:
        prompt: 输入提示词
        model_name: 模型名称，建议使用:
                   - "qwen-plus" (推荐，平衡性能)
                   - "qwen-max-latest" (最新最强，可能包括3.5)
                   - "qwen-turbo" (快速便宜)
        api_key: API密钥，如果为None则从环境变量DASHSCOPE_API_KEY读取
        api_url: API端点
        **kwargs: 其他参数 (temperature, max_tokens等)
    
    Returns:
        (响应字典, 输出文本)
    """
    # 获取API密钥
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set DASHSCOPE_API_KEY environment variable or pass api_key parameter.")
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **kwargs
    }
    
    # 发送请求
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    output_text = result['choices'][0]['message']['content']
    
    return result, output_text


def get_player_knowledge_base() -> str:
    """
    获取所有玩家的先导知识 - 行为特征描述
    """
    knowledge = """
# Rock-Paper-Scissors Player Behavior Knowledge Base

## Distribution Strategy Players
These players always choose the same action:

**A - Pure Scissors**: Always plays Scissors (0% Rock, 0% Paper, 100% Scissors)
- Completely predictable
- Will lose to Rock players, beat Paper players, tie with Scissors players

**B - Pure Rock**: Always plays Rock (100% Rock, 0% Paper, 0% Scissors)
- Completely predictable
- Will beat Scissors players, lose to Paper players, tie with Rock players

**C - Pure Paper**: Always plays Paper (0% Rock, 100% Paper, 0% Scissors)
- Completely predictable
- Will beat Rock players, lose to Scissors players, tie with Paper players

**D - Uniform Random**: Equal probability for all actions (33.3% Rock, 33.3% Paper, 33.4% Scissors)
- Unpredictable order but balanced distribution
- No exploitable pattern

**E - Rock + Paper**: Only plays Rock or Paper (50% Rock, 50% Paper, 0% Scissors)
- Never plays Scissors
- Vulnerable to Scissors-heavy strategies

**F - Rock + Scissors**: Only plays Rock or Scissors (50% Rock, 0% Paper, 50% Scissors)
- Never plays Paper
- Vulnerable to Paper-heavy strategies

**G - Paper + Scissors**: Only plays Paper or Scissors (0% Rock, 50% Paper, 50% Scissors)
- Never plays Rock
- Vulnerable to Rock-heavy strategies

**H - Rock Biased**: Prefers Rock (50% Rock, 25% Paper, 25% Scissors)
- Rock appears twice as often as other moves
- Can be exploited by Paper-biased strategies

**I - Paper Biased**: Prefers Paper (25% Rock, 50% Paper, 25% Scissors)
- Paper appears twice as often as other moves
- Can be exploited by Scissors-biased strategies

**J - Scissors Biased**: Prefers Scissors (25% Rock, 25% Paper, 50% Scissors)
- Scissors appears twice as often as other moves
- Can be exploited by Rock-biased strategies

**K - Rock > Paper**: Strong Rock preference (50% Rock, 33.3% Paper, 16.7% Scissors)
- Rock most common, Paper second, Scissors least
- Hierarchy: Rock > Paper > Scissors

**L - Rock > Scissors**: Strong Rock preference (50% Rock, 16.7% Paper, 33.3% Scissors)
- Rock most common, Scissors second, Paper least
- Hierarchy: Rock > Scissors > Paper

**M - Paper > Rock**: Strong Paper preference (33.3% Rock, 50% Paper, 16.7% Scissors)
- Paper most common, Rock second, Scissors least
- Hierarchy: Paper > Rock > Scissors

**N - Paper > Scissors**: Strong Paper preference (16.7% Rock, 50% Paper, 33.3% Scissors)
- Paper most common, Scissors second, Rock least
- Hierarchy: Paper > Scissors > Rock

**O - Scissors > Rock**: Strong Scissors preference (33.3% Rock, 16.7% Paper, 50% Scissors)
- Scissors most common, Rock second, Paper least
- Hierarchy: Scissors > Rock > Paper

**P - Scissors > Paper**: Strong Scissors preference (16.7% Rock, 33.3% Paper, 50% Scissors)
- Scissors most common, Paper second, Rock least
- Hierarchy: Scissors > Paper > Rock

## Markov Based Players 
These players adapt their strategy based on opponent's previous moves:

**X - Win-Last**: Plays the move that would have beaten opponent's last move
- First move: Random
- Subsequent moves: Counter opponent's previous move
- Against Pure Rock (always Rock): Will play Paper after first round
- Highly effective against predictable opponents

**Y - Lose-Last**: Plays the move that would have lost to opponent's last move
- First move: Random
- Subsequent moves: Intentionally lose to opponent's previous move
- Against Pure Rock: Will keep playing Scissors
- Intentionally losing strategy

**Z - Copy-Last**: Copies opponent's last move
- First move: Random
- Subsequent moves: Mirrors opponent's previous move
- Against Pure Rock: Will always play Rock (all ties)
- Creates mirror matches

## Key Insights:
1. Pure strategies are completely predictable and can be easily exploited
2. Distribution strategies maintain consistent ratios but unpredictable order
3. Reactive strategies are most effective against predictable opponents
4. Win-Last (X) tends to perform best overall
5. Strategies with heavy bias can be counter-strategized
"""
    return knowledge


def analyze_game_trajectory(player1_id: str, player2_id: str, 
                            player1_trajectory: str, player2_trajectory: str,
                            player1_wins: int, player2_wins: int, draws: int,
                            num_rounds: int) -> Dict[str, Any]:
    """
    使用LLM分析游戏轨迹
    
    Args:
        player1_id: 玩家1 ID
        player2_id: 玩家2 ID
        player1_trajectory: 玩家1轨迹
        player2_trajectory: 玩家2轨迹
        player1_wins: 玩家1胜场
        player2_wins: 玩家2胜场
        draws: 平局数
        num_rounds: 总回合数
    
    Returns:
        包含分析结果的字典
    """
    knowledge_base = get_player_knowledge_base()
    
    prompt = f"""{knowledge_base}

## Game Analysis Task

You are a Rock-Paper-Scissors strategy expert. Above is the background knowledge of 19 player strategies.

Now analyze a game where both players' identities are unknown based on their move trajectories:

**Game Info**:
- Total Rounds: {num_rounds}
- Results: Player1 won {player1_wins}, Player2 won {player2_wins}, Draws {draws}

**Player1 Trajectory**:
{player1_trajectory}

**Player2 Trajectory**:
{player2_trajectory}

**CRITICAL CONSTRAINTS**:
- Valid player identities are ONLY: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P (distribution strategies) and X, Y, Z (Markov/reactive strategies)
- You MUST only use these 19 letters. Do NOT invent other letters like U, T, etc.

**IMPORTANT**: You must respond with ONLY a valid JSON object, no other text. Follow this exact structure:

{{
  "has_markov_player": true/false,
  "markov_player": "player1" or "player2" or "both" or "neither",
  "player1_identity": "single letter from A-P or X-Z only",
  "player2_identity": "single letter from A-P or X-Z only",
  "reasoning": {{
    "player1_features": "observed characteristics",
    "player2_features": "observed characteristics",
    "key_evidence": "reasoning process"
  }},
  "prediction_next_100_rounds": {{
    "player1": {{"rock": 0-100, "paper": 0-100, "scissors": 0-100}},
    "player2": {{"rock": 0-100, "paper": 0-100, "scissors": 0-100}}
  }}
}}

Example: {{"player1_identity": "A", "player2_identity": "D"}}

Respond with ONLY the JSON object, nothing else.
"""
    
    print(f"\n{'='*80}")
    print("Sending request to LLM for analysis...")
    print(f"{'='*80}\n")
    
    try:
        response, output_text = get_response(prompt)
        
        return {
            "success": True,
            "player1_id": player1_id,
            "player2_id": player2_id,
            "analysis": output_text,
            "response_metadata": {
                "model": response.get("model"),
                "usage": response.get("usage"),
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "player1_id": player1_id,
            "player2_id": player2_id,
        }
