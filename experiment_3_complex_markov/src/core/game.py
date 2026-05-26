"""
游戏模拟模块
处理剪刀石头布游戏的模拟逻辑
"""
from typing import Tuple, List
from .players import Player, Action, create_player


class GameResult:
    """游戏结果"""
    
    def __init__(self, player1_name: str, player2_name: str, rounds: int):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.rounds = rounds
        self.player1_trajectory: List[Action] = []
        self.player2_trajectory: List[Action] = []
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
    
    def add_round(self, p1_action: Action, p2_action: Action, winner: int):
        """添加一回合的结果"""
        self.player1_trajectory.append(p1_action)
        self.player2_trajectory.append(p2_action)
        
        if winner == 1:
            self.player1_wins += 1
        elif winner == 2:
            self.player2_wins += 1
        else:
            self.draws += 1
    
    def get_trajectory_string(self, player_num: int) -> str:
        """获取玩家的trajectory字符串"""
        trajectory = self.player1_trajectory if player_num == 1 else self.player2_trajectory
        return " ".join([action.value for action in trajectory])
    
    def __str__(self) -> str:
        """格式化输出结果"""
        result = f"\n{'='*60}\n"
        result += f"Game Result: {self.player1_name} vs {self.player2_name}\n"
        result += f"Total Rounds: {self.rounds}\n"
        result += f"{'='*60}\n\n"
        
        result += f"{self.player1_name} trajectory:\n"
        result += f"{self.get_trajectory_string(1)}\n\n"
        
        result += f"{self.player2_name} trajectory:\n"
        result += f"{self.get_trajectory_string(2)}\n\n"
        
        result += f"{'='*60}\n"
        result += f"Final Score:\n"
        result += f"  {self.player1_name}: {self.player1_wins} wins\n"
        result += f"  {self.player2_name}: {self.player2_wins} wins\n"
        result += f"  Draws: {self.draws}\n"
        result += f"{'='*60}\n"
        
        return result


class Game:
    """剪刀石头布游戏"""
    
    @staticmethod
    def determine_winner(action1: Action, action2: Action) -> int:
        """
        判断胜负
        返回: 1 (玩家1赢), 2 (玩家2赢), 0 (平局)
        """
        if action1 == action2:
            return 0
        
        if (action1 == Action.ROCK and action2 == Action.SCISSORS) or \
           (action1 == Action.PAPER and action2 == Action.ROCK) or \
           (action1 == Action.SCISSORS and action2 == Action.PAPER):
            return 1
        
        return 2
    
    @staticmethod
    def simulate(player1_id: str, player2_id: str, num_rounds: int) -> GameResult:
        """
        模拟游戏
        
        Args:
            player1_id: 玩家1的ID (A-Z)
            player2_id: 玩家2的ID (A-Z)
            num_rounds: 游戏回合数
        
        Returns:
            GameResult对象，包含完整的游戏轨迹和结果
        """
        # 创建玩家
        player1 = create_player(player1_id)
        player2 = create_player(player2_id)
        
        # 让玩家准备游戏（生成预定义序列）
        player1.prepare(num_rounds)
        player2.prepare(num_rounds)
        
        # 创建结果对象
        result = GameResult(f"{player1_id} ({player1.name})", 
                          f"{player2_id} ({player2.name})", 
                          num_rounds)
        
        # 模拟每一回合
        for round_num in range(num_rounds):
            # 玩家选择动作
            action1 = player1.choose_action()
            action2 = player2.choose_action()
            
            # 判断胜负
            winner = Game.determine_winner(action1, action2)
            
            # 更新历史
            player1.update_history(action1, action2)
            player2.update_history(action2, action1)
            
            # 记录结果
            result.add_round(action1, action2, winner)
        
        return result
