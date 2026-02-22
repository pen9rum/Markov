"""
批量实验脚本
从有效组合中抽取10组进行LLM分析实验
"""
import json
import random
import os
import sys
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import Game
from core.players import PLAYER_CONFIGS


# 实验结果保存目录
EXPERIMENT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'experiment_results')

# Markov玩家列表
MARKOV_PLAYERS = {'X', 'Y', 'Z'}
# 非Markov玩家列表
NON_MARKOV_PLAYERS = {k for k in PLAYER_CONFIGS.keys() if k not in MARKOV_PLAYERS}


def generate_valid_combinations():
    """
    生成所有有效的玩家组合
    排除两个都是Markov的情况
    """
    all_players = list(PLAYER_CONFIGS.keys())
    
    # 类型1: 都是非Markov
    non_markov_combos = []
    for p1 in NON_MARKOV_PLAYERS:
        for p2 in NON_MARKOV_PLAYERS:
            if p1 != p2:  # 排除自己对战自己
                non_markov_combos.append((p1, p2))
    
    # 类型2: 有一个是Markov
    one_markov_combos = []
    for markov in MARKOV_PLAYERS:
        for non_markov in NON_MARKOV_PLAYERS:
            one_markov_combos.append((markov, non_markov))
            one_markov_combos.append((non_markov, markov))
    
    return non_markov_combos, one_markov_combos


def select_combinations(combo_type: int, count: int = 10):
    """
    从指定类型中随机抽取组合
    
    Args:
        combo_type: 1=都是非Markov, 2=有一个是Markov
        count: 抽取数量，默认10
    
    Returns:
        选中的组合列表
    """
    non_markov_combos, one_markov_combos = generate_valid_combinations()
    
    if combo_type == 1:
        pool = non_markov_combos
        print(f"\n总共有 {len(pool)} 种非Markov组合")
    else:
        pool = one_markov_combos
        print(f"\n总共有 {len(pool)} 种单Markov组合")
    
    if len(pool) < count:
        print(f"警告: 可用组合只有 {len(pool)} 种，少于请求的 {count} 种")
        return random.sample(pool, len(pool))
    
    return random.sample(pool, count)


def run_single_experiment(player1_id: str, player2_id: str, num_rounds: int, 
                         model_choice: str, model_name: str = None):
    """
    运行单个实验
    
    Returns:
        实验结果字典
    """
    print(f"\n{'='*80}")
    print(f"实验: {player1_id} vs {player2_id}")
    print(f"{'='*80}")
    
    # 运行游戏
    result = Game.simulate(player1_id, player2_id, num_rounds)
    
    # 获取轨迹
    player1_trajectory = result.get_trajectory_string(1)
    player2_trajectory = result.get_trajectory_string(2)
    
    # LLM分析
    analysis_result = None
    
    if model_choice == "5":
        # 云端API
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
        from main import analyze_game_trajectory_local
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
    
    # 构建实验结果
    experiment_data = {
        "ground_truth": {
            "player1_id": player1_id,
            "player2_id": player2_id,
            "player1_name": PLAYER_CONFIGS[player1_id][0],
            "player2_name": PLAYER_CONFIGS[player2_id][0],
            "player1_type": PLAYER_CONFIGS[player1_id][4].value,
            "player2_type": PLAYER_CONFIGS[player2_id][4].value,
        },
        "game_info": {
            "num_rounds": num_rounds,
            "player1_wins": result.player1_wins,
            "player2_wins": result.player2_wins,
            "draws": result.draws
        },
        "trajectories": {
            "player1": player1_trajectory,
            "player2": player2_trajectory
        },
        "llm_analysis": None,
        "success": False
    }
    
    if analysis_result and analysis_result.get('success'):
        experiment_data['success'] = True
        # 尝试解析JSON
        try:
            llm_output = json.loads(analysis_result['analysis'])
            experiment_data['llm_analysis'] = llm_output
        except json.JSONDecodeError:
            # 如果不是JSON，保存原始文本
            experiment_data['llm_analysis'] = {
                "raw_output": analysis_result['analysis']
            }
    else:
        experiment_data['error'] = analysis_result.get('error', 'Unknown') if analysis_result else 'No result'
    
    return experiment_data


def save_experiment_results(experiments: list, metadata: dict):
    """
    保存实验结果到JSON文件
    
    Args:
        experiments: 实验结果列表
        metadata: 实验元数据
    """
    # 构建目录结构: experiment_results/模型名称/with_markov或non_markov/
    model_name = metadata['model'].replace('/', '_').replace('\\', '_')  # 处理路径分隔符
    markov_type = "non_markov" if metadata['combo_type'] == 1 else "with_markov"
    
    output_dir = os.path.join(EXPERIMENT_OUTPUT_DIR, model_name, markov_type)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    output_data = {
        "metadata": metadata,
        "experiments": experiments
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n实验结果已保存到: {filepath}")
    return filepath


def main():
    """主函数"""
    print("\n" + "="*80)
    print("批量实验系统")
    print("="*80)
    
    # 选择组合类型
    print("\n选择实验组合类型:")
    print("1. 都是非Markov玩家 (A-P vs A-P)")
    print("2. 有一个Markov玩家 (X-Z vs A-P 或 A-P vs X-Z)")
    
    combo_type = input("\n选择 (1-2): ").strip()
    if combo_type not in ['1', '2']:
        print("无效的选择！")
        return
    
    combo_type_int = int(combo_type)
    combo_type_name = "non_markov_only" if combo_type_int == 1 else "one_markov"
    
    # 选择抽取数量
    count_input = input("\n抽取组合数量 (默认10): ").strip()
    count = int(count_input) if count_input else 10
    
    # 选择回合数
    rounds_input = input("每组游戏回合数 (默认100): ").strip()
    num_rounds = int(rounds_input) if rounds_input else 100
    
    # 选择LLM模型
    print("\n选择LLM模型:")
    print("1. Qwen2.5-1.5B (本地，最快，CPU可跑)")
    print("2. Qwen2.5-3B (本地，平衡)")
    print("3. Qwen2.5-7B (本地，推荐，需要GPU)")
    print("4. 自定义本地模型名称")
    print("5. Qwen云端API (需要API key)")
    
    model_choice = input("\n选择 (1-5): ").strip()
    
    model_name = None
    if model_choice not in ['1', '2', '3', '4', '5']:
        print("无效的选择，使用默认: Qwen2.5-1.5B")
        model_choice = "1"
    
    if model_choice in ['1', '2', '3']:
        model_map = {
            "1": "Qwen/Qwen2.5-1.5B-Instruct",
            "2": "Qwen/Qwen2.5-3B-Instruct",
            "3": "Qwen/Qwen2.5-7B-Instruct",
        }
        model_name = model_map[model_choice]
    elif model_choice == "4":
        model_name = input("输入模型名称: ").strip()
    else:
        model_name = "Qwen API"
    
    # 抽取组合
    print(f"\n正在抽取 {count} 组组合...")
    combinations = select_combinations(combo_type_int, count)
    
    print(f"\n选中的组合:")
    for i, (p1, p2) in enumerate(combinations, 1):
        print(f"{i}. {p1} vs {p2}")
    
    # 确认开始
    confirm = input(f"\n确认开始实验？(y/n): ").strip().lower()
    if confirm != 'y':
        print("实验已取消")
        return
    
    # 运行实验
    print(f"\n{'='*80}")
    print(f"开始批量实验 (共 {len(combinations)} 组)")
    print(f"{'='*80}")
    
    experiments = []
    for i, (player1_id, player2_id) in enumerate(combinations, 1):
        print(f"\n进度: {i}/{len(combinations)}")
        try:
            exp_result = run_single_experiment(
                player1_id, player2_id, num_rounds, 
                model_choice, model_name
            )
            experiments.append(exp_result)
            
            # 显示简要结果
            if exp_result['success']:
                llm_pred = exp_result['llm_analysis']
                p1_pred = llm_pred.get('player1_identity', 'Unknown')
                p2_pred = llm_pred.get('player2_identity', 'Unknown')
                print(f"✓ 完成: 真实({player1_id},{player2_id}) -> LLM预测({p1_pred},{p2_pred})")
            else:
                print(f"✗ 失败: {exp_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"✗ 异常: {str(e)}")
            experiments.append({
                "ground_truth": {
                    "player1_id": player1_id,
                    "player2_id": player2_id
                },
                "success": False,
                "error": str(e)
            })
    
    # 保存结果
    metadata = {
        "combo_type": combo_type_int,
        "combo_type_name": combo_type_name,
        "total_experiments": len(combinations),
        "num_rounds": num_rounds,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "success_count": sum(1 for exp in experiments if exp['success']),
        "failure_count": sum(1 for exp in experiments if not exp['success'])
    }
    
    filepath = save_experiment_results(experiments, metadata)
    
    # 统计结果
    print(f"\n{'='*80}")
    print("实验完成")
    print(f"{'='*80}")
    print(f"总实验数: {metadata['total_experiments']}")
    print(f"成功: {metadata['success_count']}")
    print(f"失败: {metadata['failure_count']}")
    print(f"结果保存至: {filepath}")


if __name__ == "__main__":
    main()
