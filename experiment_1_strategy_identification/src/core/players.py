"""
玩家策略定义模块
定义各种剪刀石头布玩家策略
"""
import random
from typing import List, Optional
from enum import Enum


class Action(Enum):
    """游戏动作"""
    ROCK = "Rock"
    PAPER = "Paper"
    SCISSORS = "Scissors"


class PlayerType(Enum):
    """玩家类型"""
    STATIC = "S"  # 静态策略
    DISTRIBUTION = "D"  # 分布策略
    HISTORY = "H"  # 历史反应策略


class Player:
    """玩家基类"""
    
    def __init__(self, name: str, player_type: PlayerType, 
                 rock_prob: float, paper_prob: float, scissors_prob: float):
        self.name = name
        self.player_type = player_type
        self.rock_prob = rock_prob
        self.paper_prob = paper_prob
        self.scissors_prob = scissors_prob
        self.history: List[Action] = []
        self.opponent_history: List[Action] = []
    
    def prepare(self, num_rounds: int):
        """准备游戏 - 在游戏开始前调用"""
        pass
    
    def choose_action(self) -> Action:
        """选择一个动作"""
        raise NotImplementedError
    
    def update_history(self, my_action: Action, opponent_action: Action):
        """更新历史记录"""
        self.history.append(my_action)
        self.opponent_history.append(opponent_action)
    
    def reset(self):
        """重置历史"""
        self.history = []
        self.opponent_history = []


class StaticPlayer(Player):
    """静态玩家 - 总是出相同的动作"""
    
    def choose_action(self) -> Action:
        if self.rock_prob == 1.0:
            return Action.ROCK
        elif self.paper_prob == 1.0:
            return Action.PAPER
        else:
            return Action.SCISSORS


class DistributionPlayer(Player):
    """分布玩家 - 按照精确概率分布，位置随机"""
    
    def __init__(self, name: str, player_type: PlayerType,
                 rock_prob: float, paper_prob: float, scissors_prob: float):
        super().__init__(name, player_type, rock_prob, paper_prob, scissors_prob)
        self.action_sequence: List[Action] = []
        self.current_index = 0
    
    def prepare(self, num_rounds: int):
        """根据概率分布生成精确数量的动作序列"""
        # 计算每种动作的精确数量
        rock_count = round(num_rounds * self.rock_prob)
        paper_count = round(num_rounds * self.paper_prob)
        scissors_count = num_rounds - rock_count - paper_count  # 确保总数正确
        
        # 处理round导致的负数问题
        # 如果scissors_count为负，从paper_count或rock_count中减去
        if scissors_count < 0:
            if paper_count > 0:
                paper_count += scissors_count
                scissors_count = 0
            elif rock_count > 0:
                rock_count += scissors_count
                scissors_count = 0
        
        # 生成动作序列
        self.action_sequence = (
            [Action.ROCK] * max(0, rock_count) +
            [Action.PAPER] * max(0, paper_count) +
            [Action.SCISSORS] * max(0, scissors_count)
        )
        
        # 随机打乱顺序
        random.shuffle(self.action_sequence)
        self.current_index = 0
    
    def choose_action(self) -> Action:
        """按照预定义序列返回动作"""
        if self.current_index >= len(self.action_sequence):
            # 如果序列用完了，返回随机动作（通常不应该发生）
            return random.choice([Action.ROCK, Action.PAPER, Action.SCISSORS])
        
        action = self.action_sequence[self.current_index]
        self.current_index += 1
        return action
    
    def reset(self):
        """重置状态"""
        super().reset()
        self.action_sequence = []
        self.current_index = 0


class ReactivePlayer(Player):
    """反应式玩家 - 基于对手上一回合的动作做出反应"""
    
    def __init__(self, name: str, player_type: PlayerType,
                 rock_prob: float, paper_prob: float, scissors_prob: float,
                 strategy: str):
        super().__init__(name, player_type, rock_prob, paper_prob, scissors_prob)
        self.strategy = strategy  # "Win-Last", "Lose-Last", "Copy-Last"
    
    def choose_action(self) -> Action:
        # 如果没有历史，使用随机策略
        if not self.opponent_history:
            rand = random.random()
            if rand < 0.333:
                return Action.ROCK
            elif rand < 0.666:
                return Action.PAPER
            else:
                return Action.SCISSORS
        
        last_opponent_action = self.opponent_history[-1]
        
        if self.strategy == "Win-Last":
            # 出能赢对手上一回合的手势
            return self._get_winning_action(last_opponent_action)
        elif self.strategy == "Lose-Last":
            # 出会输给对手上一回合的手势
            return self._get_losing_action(last_opponent_action)
        else:  # Copy-Last
            # 复制对手上一回合的手势
            return last_opponent_action
    
    @staticmethod
    def _get_winning_action(opponent_action: Action) -> Action:
        """获取能赢过对手动作的手势"""
        if opponent_action == Action.ROCK:
            return Action.PAPER
        elif opponent_action == Action.PAPER:
            return Action.SCISSORS
        else:  # SCISSORS
            return Action.ROCK
    
    @staticmethod
    def _get_losing_action(opponent_action: Action) -> Action:
        """获取会输给对手动作的手势"""
        if opponent_action == Action.ROCK:
            return Action.SCISSORS
        elif opponent_action == Action.PAPER:
            return Action.ROCK
        else:  # SCISSORS
            return Action.PAPER


# 预定义的玩家配置
PLAYER_CONFIGS = {
    "A": ("Pure Scissors", 0, 0, 1, PlayerType.STATIC, None),
    "B": ("Pure Rock", 1, 0, 0, PlayerType.STATIC, None),
    "C": ("Pure Paper", 0, 1, 0, PlayerType.STATIC, None),
    "D": ("Uniform Random", 0.333, 0.333, 0.334, PlayerType.DISTRIBUTION, None),
    "E": ("Rock + Paper", 0.50, 0.50, 0, PlayerType.DISTRIBUTION, None),
    "F": ("Rock + Scissors", 0.50, 0, 0.50, PlayerType.DISTRIBUTION, None),
    "G": ("Paper + Scissors", 0, 0.50, 0.50, PlayerType.DISTRIBUTION, None),
    "H": ("Rock Biased", 0.50, 0.25, 0.25, PlayerType.DISTRIBUTION, None),
    "I": ("Paper Biased", 0.25, 0.50, 0.25, PlayerType.DISTRIBUTION, None),
    "J": ("Scissors Biased", 0.25, 0.25, 0.50, PlayerType.DISTRIBUTION, None),
    "K": ("Rock > Paper", 0.50, 0.333, 0.167, PlayerType.DISTRIBUTION, None),
    "L": ("Rock > Scissors", 0.50, 0.167, 0.333, PlayerType.DISTRIBUTION, None),
    "M": ("Paper > Rock", 0.333, 0.50, 0.167, PlayerType.DISTRIBUTION, None),
    "N": ("Paper > Scissors", 0.167, 0.50, 0.333, PlayerType.DISTRIBUTION, None),
    "O": ("Scissors > Rock", 0.333, 0.167, 0.50, PlayerType.DISTRIBUTION, None),
    "P": ("Scissors > Paper", 0.167, 0.333, 0.50, PlayerType.DISTRIBUTION, None),
    "X": ("Win-Last", 0, 0, 0, PlayerType.HISTORY, "Win-Last"),
    "Y": ("Lose-Last", 0, 0, 0, PlayerType.HISTORY, "Lose-Last"),
    "Z": ("Copy-Last", 0, 0, 0, PlayerType.HISTORY, "Copy-Last"),
}


def create_player(player_id: str) -> Player:
    """根据玩家ID创建玩家实例"""
    if player_id not in PLAYER_CONFIGS:
        raise ValueError(f"Unknown player ID: {player_id}")
    
    name, rock, paper, scissors, ptype, strategy = PLAYER_CONFIGS[player_id]
    
    if ptype == PlayerType.STATIC:
        return StaticPlayer(name, ptype, rock, paper, scissors)
    elif ptype == PlayerType.DISTRIBUTION:
        return DistributionPlayer(name, ptype, rock, paper, scissors)
    else:  # HISTORY
        return ReactivePlayer(name, ptype, rock, paper, scissors, strategy)
