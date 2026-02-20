"""
主程序入口
运行剪刀石头布游戏模拟，支持本地LLM和云端API分析
"""
import json
import os
from datetime import datetime
from core.game import Game
from core.players import PLAYER_CONFIGS

# 分析结果保存目录
ANALYSIS_OUTPUT_DIR = "analysis_results"


def print_available_players():
    """打印所有可用的玩家"""
    print("\n可用的玩家列表:")
    print("="*70)
    for player_id, config in sorted(PLAYER_CONFIGS.items()):
        name, rock, paper, scissors, ptype, strategy = config
        print(f"{player_id}: {name:25} | R:{rock:.3f} P:{paper:.3f} S:{scissors:.3f} | {ptype.value}")
    print("="*70)


def get_player_knowledge_base() -> str:
    """获取所有玩家的先导知识"""
    from analysis.llm import get_player_knowledge_base
    return get_player_knowledge_base()


def analyze_game_trajectory_local(player1_id: str, player2_id: str,
                                  player1_trajectory: str, player2_trajectory: str,
                                  player1_wins: int, player2_wins: int, draws: int,
                                  num_rounds: int,
                                  model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """使用本地模型分析游戏轨迹"""
    from analysis.llm_local import get_response_local
    
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

**Player {player1_id} Trajectory**:
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
    print(f"使用本地模型分析: {model_name}")
    print(f"{'='*80}\n")
    
    try:
        metadata, output_text = get_response_local(
            prompt,
            model_name=model_name,
            max_length=2048
        )
        
        return {
            "success": True,
            "player1_id": player1_id,
            "player2_id": player2_id,
            "analysis": output_text,
            "response_metadata": metadata
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "player1_id": player1_id,
            "player2_id": player2_id,
        }


def save_analysis_to_file(analysis_result: dict, filename: str = None):
    """保存分析结果到文件"""
    # 确保输出目录存在
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{analysis_result['player1_id']}_vs_{analysis_result['player2_id']}_{timestamp}.txt"
    
    # 完整路径
    filepath = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM Game Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        if analysis_result['success']:
            f.write(f"Match: {analysis_result['player1_id']} vs {analysis_result['player2_id']}\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("LLM Analysis:\n")
            f.write("-"*80 + "\n\n")
            f.write(analysis_result['analysis'])
            f.write("\n\n")
            
            f.write("-"*80 + "\n")
            f.write("Metadata:\n")
            f.write("-"*80 + "\n")
            if 'response_metadata' in analysis_result:
                metadata = analysis_result['response_metadata']
                for key, value in metadata.items():
                    f.write(f"{key.capitalize()}: {value}\n")
        else:
            f.write(f"Analysis Failed\n")
            f.write(f"Error: {analysis_result.get('error', 'Unknown')}\n")
    
    return filename


def main():
    """主函数"""
    print("\n欢迎使用剪刀石头布模拟系统！")
    print_available_players()
    
    # 获取用户输入
    print("\n请输入游戏参数:")
    player1_id = input("玩家1 ID (A-Z): ").strip().upper()
    player2_id = input("玩家2 ID (A-Z): ").strip().upper()
    
    try:
        num_rounds = int(input("游戏回合数: ").strip())
        if num_rounds <= 0:
            print("回合数必须是正整数！")
            return
    except ValueError:
        print("无效的回合数！")
        return
    
    # 验证玩家ID
    if player1_id not in PLAYER_CONFIGS:
        print(f"无效的玩家1 ID: {player1_id}")
        return
    
    if player2_id not in PLAYER_CONFIGS:
        print(f"无效的玩家2 ID: {player2_id}")
        return
    
    # 运行模拟
    print(f"\n开始模拟 {num_rounds} 回合的游戏...")
    print(f"玩家1: {player1_id} ({PLAYER_CONFIGS[player1_id][0]})")
    print(f"玩家2: {player2_id} ({PLAYER_CONFIGS[player2_id][0]})")
    print(f"{'='*80}\n")
    
    result = Game.simulate(player1_id, player2_id, num_rounds)
    
    # 输出结果
    print(result)
    
    # 询问是否进行LLM分析
    print("\n是否使用LLM分析此局游戏？(y/n): ", end="")
    if input().strip().lower() == 'y':
        
        # 选择模型
        print("\n选择LLM模型:")
        print("1. Qwen2.5-1.5B (本地，最快，CPU可跑)")
        print("2. Qwen2.5-3B (本地，平衡)")
        print("3. Qwen2.5-7B (本地，推荐，需要GPU)")
        print("4. 自定义本地模型名称")
        print("5. Qwen云端API (需要API key)")
        
        choice = input("选择 (1-5): ").strip()
        
        # 获取轨迹
        player1_trajectory = result.get_trajectory_string(1)
        player2_trajectory = result.get_trajectory_string(2)
        
        analysis_result = None
        
        if choice == "5":
            # 云端API
            print("\n使用Qwen云端API...")
            from analysis.llm import analyze_game_trajectory
            
            analysis_result = analyze_game_trajectory(
                player1_id=player1_id,
                player2_id=player2_id,
                player1_trajectory=player1_trajectory,
                player2_trajectory=player2_trajectory,
                player1_wins=result.player1_wins,
                player2_wins=result.player2_wins,
                draws=result.draws,
                num_rounds=num_rounds
            )
        else:
            # 本地模型
            model_map = {
                "1": "Qwen/Qwen2.5-1.5B-Instruct",
                "2": "Qwen/Qwen2.5-3B-Instruct",
                "3": "Qwen/Qwen2.5-7B-Instruct",
            }
            
            if choice in model_map:
                model_name = model_map[choice]
            elif choice == "4":
                model_name = input("输入模型名称: ").strip()
            else:
                model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            
            print(f"\n正在使用本地模型 {model_name} 进行分析...")
            
            analysis_result = analyze_game_trajectory_local(
                player1_id=player1_id,
                player2_id=player2_id,
                player1_trajectory=player1_trajectory,
                player2_trajectory=player2_trajectory,
                player1_wins=result.player1_wins,
                player2_wins=result.player2_wins,
                draws=result.draws,
                num_rounds=num_rounds,
                model_name=model_name
            )
        
        # 显示分析结果
        if analysis_result and analysis_result['success']:
            print("\n" + "="*80)
            print("LLM分析结果:")
            print("="*80 + "\n")
            print(analysis_result['analysis'])
            
            # 显示元数据
            if 'response_metadata' in analysis_result:
                metadata = analysis_result['response_metadata']
                if 'total_tokens' in metadata:
                    print(f"\nToken使用: {metadata['total_tokens']}")
                if 'device' in metadata:
                    print(f"设备: {metadata['device']}")
                if 'model' in metadata:
                    print(f"模型: {metadata['model']}")
            
            # 询问是否保存
            print("\n是否保存分析结果到文件？(y/n): ", end="")
            if input().strip().lower() == 'y':
                filename = save_analysis_to_file(analysis_result)
                print(f"\n分析结果已保存到: {filename}")
        elif analysis_result:
            print(f"\nLLM分析失败: {analysis_result.get('error', 'Unknown error')}")
            if choice != "5":
                print("\n提示: 确保已安装 transformers 和 torch")
                print("运行: pip install transformers torch")
            else:
                print("\n提示: 检查DASHSCOPE_API_KEY环境变量是否设置")
        else:
            print("\n未进行LLM分析")
    
    # 输出结果
    print(result)
    
    # 询问是否继续
    print("\n是否继续模拟？(y/n): ", end="")
    if input().strip().lower() == 'y':
        main()
    else:
        print("\n感谢使用！再见！")


if __name__ == "__main__":
    main()
