"""
验证所有玩家的分布是否正确
"""
import sys
import os

# 添加父目录到路径以便导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.game import Game
from src.core.players import PLAYER_CONFIGS, Action


def verify_player_distribution(player_id: str, num_rounds: int):
    """验证单个玩家的分布"""
    config = PLAYER_CONFIGS[player_id]
    name, expected_rock, expected_paper, expected_scissors, ptype, _ = config
    
    # 对战一个不会影响其策略的玩家
    result = Game.simulate(player_id, "A", num_rounds)
    trajectory = result.player1_trajectory
    
    # 统计实际分布
    rock_count = sum(1 for action in trajectory if action == Action.ROCK)
    paper_count = sum(1 for action in trajectory if action == Action.PAPER)
    scissors_count = sum(1 for action in trajectory if action == Action.SCISSORS)
    
    # 计算期望值（与实际代码逻辑一致）
    expected_rock_count = round(num_rounds * expected_rock)
    expected_paper_count = round(num_rounds * expected_paper)
    expected_scissors_count = num_rounds - expected_rock_count - expected_paper_count
    
    # 处理round导致的负数问题（与players.py中的逻辑一致）
    if expected_scissors_count < 0:
        if expected_paper_count > 0:
            expected_paper_count += expected_scissors_count
            expected_scissors_count = 0
        elif expected_rock_count > 0:
            expected_rock_count += expected_scissors_count
            expected_scissors_count = 0
    
    # 检查是否匹配
    rock_match = rock_count == expected_rock_count
    paper_match = paper_count == expected_paper_count
    scissors_match = scissors_count == expected_scissors_count
    
    all_match = rock_match and paper_match and scissors_match
    
    return {
        'player_id': player_id,
        'name': name,
        'expected': (expected_rock_count, expected_paper_count, expected_scissors_count),
        'actual': (rock_count, paper_count, scissors_count),
        'match': all_match
    }


def verify_all_distributions(num_rounds: int = 100):
    """验证所有玩家的分布"""
    print(f"\n{'='*90}")
    print(f"验证所有玩家分布是否正确 ({num_rounds} 回合)")
    print(f"{'='*90}\n")
    
    # 只验证非反应式玩家（A-P），因为反应式玩家依赖对手
    static_and_dist_players = [pid for pid in PLAYER_CONFIGS.keys() 
                               if pid not in ['X', 'Y', 'Z']]
    
    all_correct = True
    incorrect_players = []
    
    print(f"{'ID':<3} {'玩家':<25} {'期望(R/P/S)':<20} {'实际(R/P/S)':<20} {'状态'}")
    print("-" * 90)
    
    for player_id in static_and_dist_players:
        result = verify_player_distribution(player_id, num_rounds)
        
        exp_str = f"{result['expected'][0]}/{result['expected'][1]}/{result['expected'][2]}"
        act_str = f"{result['actual'][0]}/{result['actual'][1]}/{result['actual'][2]}"
        status = "✓ 正确" if result['match'] else "✗ 错误"
        
        print(f"{result['player_id']:<3} {result['name']:<25} {exp_str:<20} {act_str:<20} {status}")
        
        if not result['match']:
            all_correct = False
            incorrect_players.append(result)
    
    print("-" * 90)
    
    if all_correct:
        print(f"\n✓ 所有玩家分布都正确！")
    else:
        print(f"\n✗ 发现 {len(incorrect_players)} 个玩家分布不正确：")
        for result in incorrect_players:
            print(f"  - {result['player_id']} ({result['name']}): "
                  f"期望 {result['expected']} vs 实际 {result['actual']}")
    
    print(f"\n{'='*90}\n")
    
    return all_correct


def verify_distribution_consistency(player_id: str, num_rounds: int, num_trials: int = 5):
    """验证玩家在多次试验中分布的一致性"""
    print(f"\n验证玩家 {player_id} 的一致性 ({num_trials} 次试验):")
    print("-" * 60)
    
    results = []
    for trial in range(num_trials):
        result = Game.simulate(player_id, "A", num_rounds)
        trajectory = result.player1_trajectory
        
        rock_count = sum(1 for action in trajectory if action == Action.ROCK)
        paper_count = sum(1 for action in trajectory if action == Action.PAPER)
        scissors_count = sum(1 for action in trajectory if action == Action.SCISSORS)
        
        results.append((rock_count, paper_count, scissors_count))
        print(f"试验 {trial+1}: Rock={rock_count} Paper={paper_count} Scissors={scissors_count}")
    
    # 检查是否所有试验都一致
    all_same = all(r == results[0] for r in results)
    
    if all_same:
        print(f"✓ 所有试验分布完全一致")
    else:
        print(f"✗ 试验之间分布不一致")
    
    return all_same


if __name__ == "__main__":
    print("\n" + "="*90)
    print("分布验证测试")
    print("="*90)
    
    # 测试1: 验证所有玩家在100回合的分布
    verify_all_distributions(100)
    
    # 测试2: 验证一致性
    print("\n" + "="*90)
    print("一致性测试 - 验证同一玩家多次对战是否保持相同分布")
    print("="*90)
    
    test_players = ['D', 'H', 'I', 'K']
    all_consistent = True
    
    for pid in test_players:
        consistent = verify_distribution_consistency(pid, 100, 5)
        if not consistent:
            all_consistent = False
    
    print("\n" + "="*90)
    if all_consistent:
        print("✓ 所有测试玩家在多次试验中分布保持一致")
    else:
        print("✗ 部分玩家在多次试验中分布不一致")
    print("="*90 + "\n")
    
    # 测试3: 验证不同回合数
    print("\n" + "="*90)
    print("不同回合数测试")
    print("="*90)
    
    for rounds in [33, 47, 100, 200]:
        print(f"\n--- {rounds} 回合 ---")
        verify_all_distributions(rounds)
