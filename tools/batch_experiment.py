"""
批量实验脚本
从有效组合中抽取指定组数进行LLM分析实验
"""
import json
import random
import os
import sys
import argparse
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import Game
from core.players import PLAYER_CONFIGS


# 批量实验结果保存目录
BATCH_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'batch_results')

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
    
    if model_choice in ["5", "6"]:
        # 云端API (Qwen API 或 Gemini)
        from analysis.llm import analyze_game_trajectory
        
        # 确定API类型
        if model_choice == "6":
            api_type = "gemini"
        else:
            api_type = "qwen"
        
        analysis_result = analyze_game_trajectory(
            player1_id=player1_id,
            player2_id=player2_id,
            player1_trajectory=player1_trajectory,
            player2_trajectory=player2_trajectory,
            player1_wins=result.player1_wins,
            player2_wins=result.player2_wins,
            draws=result.draws,
            num_rounds=num_rounds,
            api_type=api_type,
            model_name=model_name
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
        # 保存原始文本
        experiment_data['llm_analysis'] = {
            "raw_output": analysis_result['analysis']
        }
    else:
        experiment_data['error'] = analysis_result.get('error', 'Unknown') if analysis_result else 'No result'
    
    return experiment_data


def save_single_analysis(exp_data: dict, model_name: str, combo_type: int):
    """
    保存单个实验的分析结果为文本文件（模仿main.py的格式）
    
    Args:
        exp_data: 单个实验数据
        model_name: 模型名称
        combo_type: 组合类型 (1=非Markov, 2=含Markov)
    
    Returns:
        保存的文件路径
    """
    # 构建目录结构: batch_results/模型名称/type1_non_markov 或 type2_with_markov/
    clean_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    if combo_type == 1:
        type_folder = "type1_non_markov"
    else:
        type_folder = "type2_with_markov"
    
    output_dir = os.path.join(BATCH_OUTPUT_DIR, clean_model_name, type_folder)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 包含毫秒避免重复
    player1_id = exp_data['ground_truth']['player1_id']
    player2_id = exp_data['ground_truth']['player2_id']
    filename = f"analysis_{player1_id}_vs_{player2_id}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # 写入文本文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM Game Analysis Report (Batch Experiment)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Match: {player1_id} vs {player2_id}\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n\n")
        
        # 游戏信息
        if 'game_info' in exp_data:
            game_info = exp_data['game_info']
            f.write("-"*80 + "\n")
            f.write("Game Results\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Total Rounds: {game_info.get('num_rounds', 'N/A')}\n")
            f.write(f"Player1 Wins: {game_info.get('player1_wins', 'N/A')}\n")
            f.write(f"Player2 Wins: {game_info.get('player2_wins', 'N/A')}\n")
            f.write(f"Draws: {game_info.get('draws', 'N/A')}\n\n")
        
        # 实际轨迹和分布
        if 'trajectories' in exp_data:
            f.write("-"*80 + "\n")
            f.write("Actual Game Data\n")
            f.write("-"*80 + "\n\n")
            
            trajectories = exp_data['trajectories']
            if 'player1' in trajectories:
                f.write(f"Player1 ({player1_id}) Trajectory:\n")
                f.write(f"{trajectories['player1']}\n\n")
            
            if 'player2' in trajectories:
                f.write(f"Player2 ({player2_id}) Trajectory:\n")
                f.write(f"{trajectories['player2']}\n\n")
            
            # 计算实际分布
            if 'player1' in trajectories:
                p1_traj = trajectories['player1']
                p1_rock = p1_traj.count('R')
                p1_paper = p1_traj.count('P')
                p1_scissors = p1_traj.count('S')
                p1_total = p1_rock + p1_paper + p1_scissors
                
                if p1_total > 0:
                    f.write(f"Player1 Actual Distribution:\n")
                    f.write(f"  Rock: {p1_rock} ({p1_rock/p1_total*100:.1f}%)\n")
                    f.write(f"  Paper: {p1_paper} ({p1_paper/p1_total*100:.1f}%)\n")
                    f.write(f"  Scissors: {p1_scissors} ({p1_scissors/p1_total*100:.1f}%)\n\n")
            
            if 'player2' in trajectories:
                p2_traj = trajectories['player2']
                p2_rock = p2_traj.count('R')
                p2_paper = p2_traj.count('P')
                p2_scissors = p2_traj.count('S')
                p2_total = p2_rock + p2_paper + p2_scissors
                
                if p2_total > 0:
                    f.write(f"Player2 Actual Distribution:\n")
                    f.write(f"  Rock: {p2_rock} ({p2_rock/p2_total*100:.1f}%)\n")
                    f.write(f"  Paper: {p2_paper} ({p2_paper/p2_total*100:.1f}%)\n")
                    f.write(f"  Scissors: {p2_scissors} ({p2_scissors/p2_total*100:.1f}%)\n\n")
        
        # LLM分析结果
        f.write("-"*80 + "\n")
        f.write("LLM Analysis:\n")
        f.write("-"*80 + "\n\n")
        
        if exp_data.get('success'):
            llm_analysis = exp_data.get('llm_analysis')
            if isinstance(llm_analysis, dict) and 'raw_output' in llm_analysis:
                f.write(llm_analysis['raw_output'])
            elif isinstance(llm_analysis, str):
                f.write(llm_analysis)
            else:
                f.write(json.dumps(llm_analysis, ensure_ascii=False, indent=2))
        else:
            f.write(f"Analysis failed: {exp_data.get('error', 'Unknown error')}\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    return filepath


def save_batch_summary(experiments: list, combo_type: int, 
                       model_name: str, num_rounds: int):
    """
    保存批量实验的汇总信息
    
    Args:
        experiments: 实验结果列表
        combo_type: 组合类型
        model_name: 模型名称
        num_rounds: 游戏回合数
    
    Returns:
        保存的文件路径
    """
    clean_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    if combo_type == 1:
        type_folder = "type1_non_markov"
        type_description = "都是非Markov玩家 (A-P vs A-P)"
    else:
        type_folder = "type2_with_markov"
        type_description = "有一个Markov玩家 (X-Z vs A-P)"
    
    output_dir = os.path.join(BATCH_OUTPUT_DIR, clean_model_name, type_folder)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"_batch_summary_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    summary_data = {
        "metadata": {
            "combo_type": combo_type,
            "combo_type_name": type_folder,
            "description": type_description,
            "model": model_name,
            "num_rounds": num_rounds,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "success_count": sum(1 for exp in experiments if exp['success']),
            "failure_count": sum(1 for exp in experiments if not exp['success'])
        },
        "experiments_summary": [
            {
                "player1_id": exp['ground_truth']['player1_id'],
                "player2_id": exp['ground_truth']['player2_id'],
                "success": exp['success']
            }
            for exp in experiments
        ]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    return filepath


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量實驗系統 - 運行多組玩家對戰並進行LLM分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 運行10組類型1（非Markov）和5組類型2（含Markov），使用Gemini API
  python tools/batch_experiment.py --type1 10 --type2 5 --rounds 100 --model gemini
  
  # 只運行類型1，使用本地Qwen2.5-1.5B模型
  python tools/batch_experiment.py --type1 20 --rounds 200 --model qwen-1.5b
  
  # 使用Qwen雲端API
  python tools/batch_experiment.py --type1 5 --type2 5 --model qwen-api
        """
    )
    
    parser.add_argument('--type1', type=int, default=0,
                       help='類型1組數（都是非Markov玩家 A-P vs A-P）')
    parser.add_argument('--type2', type=int, default=0,
                       help='類型2組數（有一個Markov玩家 X-Z vs A-P）')
    parser.add_argument('--rounds', type=int, default=100,
                       help='每組遊戲回合數（默認: 100）')
    parser.add_argument('--model', type=str, default='qwen-1.5b',
                       choices=['qwen-1.5b', 'qwen-3b', 'qwen-7b', 'qwen-api', 'gemini'],
                       help='LLM模型選擇（默認: qwen-1.5b）')
    parser.add_argument('--custom-model', type=str,
                       help='自定義本地模型名稱（使用此參數時忽略--model）')
    
    args = parser.parse_args()
    
    # 檢查至少要有一種類型的實驗
    if args.type1 == 0 and args.type2 == 0:
        parser.error("至少需要指定 --type1 或 --type2 其中一個（組數 > 0）")
    
    # 確定模型名稱
    if args.custom_model:
        model_name = args.custom_model
        model_choice = "custom"
    else:
        model_map = {
            'qwen-1.5b': ('Qwen/Qwen2.5-1.5B-Instruct', '1'),
            'qwen-3b': ('Qwen/Qwen2.5-3B-Instruct', '2'),
            'qwen-7b': ('Qwen/Qwen2.5-7B-Instruct', '3'),
            'qwen-api': ('qwen-plus', '5'),
            'gemini': ('gemini-3-flash-preview', '6')
        }
        model_name, model_choice = model_map[args.model]
    
    print("\n" + "="*80)
    print("批量實驗系統")
    print("="*80)
    print(f"\n實驗配置:")
    print(f"  類型1（非Markov）: {args.type1} 組")
    print(f"  類型2（含Markov）: {args.type2} 組")
    print(f"  每組回合數: {args.rounds}")
    print(f"  LLM模型: {model_name}")
    
    # 收集所有實驗
    all_experiments = []
    all_combinations = []
    
    # 類型1: 非Markov
    if args.type1 > 0:
        print(f"\n正在抽取類型1組合...")
        type1_combos = select_combinations(1, args.type1)
        print(f"選中 {len(type1_combos)} 組類型1組合:")
        for i, (p1, p2) in enumerate(type1_combos, 1):
            print(f"  {i}. {p1} vs {p2}")
        all_combinations.extend([(1, p1, p2) for p1, p2 in type1_combos])
    
    # 類型2: 含Markov
    if args.type2 > 0:
        print(f"\n正在抽取類型2組合...")
        type2_combos = select_combinations(2, args.type2)
        print(f"選中 {len(type2_combos)} 組類型2組合:")
        for i, (p1, p2) in enumerate(type2_combos, 1):
            print(f"  {i}. {p1} vs {p2}")
        all_combinations.extend([(2, p1, p2) for p1, p2 in type2_combos])
    
    # 運行實驗
    total = len(all_combinations)
    print(f"\n{'='*80}")
    print(f"開始批量實驗 (共 {total} 組)")
    print(f"{'='*80}")
    
    for i, (combo_type, player1_id, player2_id) in enumerate(all_combinations, 1):
        print(f"\n進度: {i}/{total}")
        try:
            exp_result = run_single_experiment(
                player1_id, player2_id, args.rounds, 
                model_choice, model_name
            )
            exp_result['combo_type'] = combo_type
            all_experiments.append(exp_result)
            
            # 顯示簡要結果
            if exp_result['success']:
                llm_pred = exp_result.get('llm_analysis', {})
                if 'predictions' in llm_pred:
                    # 新格式 (有ground_truth和predictions)
                    p1_pred = llm_pred['predictions'].get('player1', {}).get('identity', 'Unknown')
                    p2_pred = llm_pred['predictions'].get('player2', {}).get('identity', 'Unknown')
                else:
                    # 舊格式
                    p1_pred = llm_pred.get('player1_identity', 'Unknown')
                    p2_pred = llm_pred.get('player2_identity', 'Unknown')
                print(f"✓ 完成: 真實({player1_id},{player2_id}) -> LLM預測({p1_pred},{p2_pred})")
            else:
                print(f"✗ 失敗: {exp_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"✗ 異常: {str(e)}")
            all_experiments.append({
                "ground_truth": {
                    "player1_id": player1_id,
                    "player2_id": player2_id
                },
                "combo_type": combo_type,
                "success": False,
                "error": str(e)
            })
    
    # 分離類型1和類型2的結果
    type1_experiments = [exp for exp in all_experiments if exp.get('combo_type') == 1]
    type2_experiments = [exp for exp in all_experiments if exp.get('combo_type') == 2]
    
    # 保存每個實驗的文本文件
    print(f"\n{'='*80}")
    print("正在保存實驗結果...")
    print(f"{'='*80}")
    
    type1_files = []
    type2_files = []
    
    for exp in type1_experiments:
        filepath = save_single_analysis(exp, model_name, combo_type=1)
        type1_files.append(filepath)
    
    for exp in type2_experiments:
        filepath = save_single_analysis(exp, model_name, combo_type=2)
        type2_files.append(filepath)
    
    # 統計結果
    print(f"\n{'='*80}")
    print("實驗完成")
    print(f"{'='*80}")
    print(f"總實驗數: {total}")
    print(f"  類型1（非Markov）: {len(type1_experiments)} 組")
    print(f"    成功: {sum(1 for exp in type1_experiments if exp['success'])}")
    print(f"    失敗: {sum(1 for exp in type1_experiments if not exp['success'])}")
    print(f"    文件數: {len(type1_files)}")
    print(f"  類型2（含Markov）: {len(type2_experiments)} 組")
    print(f"    成功: {sum(1 for exp in type2_experiments if exp['success'])}")
    print(f"    失敗: {sum(1 for exp in type2_experiments if not exp['success'])}")
    print(f"    文件數: {len(type2_files)}")


if __name__ == "__main__":
    main()
