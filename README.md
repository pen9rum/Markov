# 剪刀石头布模拟系统

一个支持多种策略的剪刀石头布（Rock-Paper-Scissors）游戏模拟系统，集成LLM分析功能。

## 功能特点

- **多种玩家策略**：支持19种不同的玩家策略
  - 纯策略（总是出同一种）
  - 概率分布策略（按照给定概率随机出招）
  - 反应式策略（基于对手历史动作）

- **完整的游戏追踪**：记录每个回合双方的出招轨迹
- **详细的统计信息**：显示胜负平局的次数统计
- **LLM智能分析**：使用大语言模型分析游戏策略和行为模式

## 玩家策略列表

### 纯策略 (Static - S)
- **A**: Pure Scissors - 总是出剪刀
- **B**: Pure Rock - 总是出石头
- **C**: Pure Paper - 总是出布

### 概率分布策略 (Distribution - D)
- **D**: Uniform Random - 均匀随机 (0.333, 0.333, 0.334)
- **E**: Rock + Paper - 只出石头或布 (0.50, 0.50, 0)
- **F**: Rock + Scissors - 只出石头或剪刀 (0.50, 0, 0.50)
- **G**: Paper + Scissors - 只出布或剪刀 (0, 0.50, 0.50)
- **H**: Rock Biased - 石头偏向 (0.50, 0.25, 0.25)
- **I**: Paper Biased - 布偏向 (0.25, 0.50, 0.25)
- **J**: Scissors Biased - 剪刀偏向 (0.25, 0.25, 0.50)
- **K**: Rock > Paper - 石头>布>剪刀 (0.50, 0.333, 0.167)
- **L**: Rock > Scissors - 石头>剪刀>布 (0.50, 0.167, 0.333)
- **M**: Paper > Rock - 布>石头>剪刀 (0.333, 0.50, 0.167)
- **N**: Paper > Scissors - 布>剪刀>石头 (0.167, 0.50, 0.333)
- **O**: Scissors > Rock - 剪刀>石头>布 (0.333, 0.167, 0.50)
- **P**: Scissors > Paper - 剪刀>布>石头 (0.167, 0.333, 0.50)

### 反应式策略 (History/Reactive - H)
- **X**: Win-Last - 出能赢对手上一回合的手势
- **Y**: Lose-Last - 出会输给对手上一回合的手势
- **Z**: Copy-Last - 复制对手上一回合的手势

## 使用方法

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

只需要安装 `requests` 库，非常轻量！

### 2. 配置API密钥（仅LLM分析需要）

使用LLM分析功能需要Qwen API密钥：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Windows CMD
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

获取API密钥：访问 [阿里云百炼平台](https://dashscope.console.aliyun.com/)

### 3. 测试API连接（可选）

```bash
python test_api.py
```

这会测试qwen-plus、qwen-turbo和qwen-max-latest模型是否可用。

详细设置说明请查看 [SETUP.md](SETUP.md)

### 运行程序

**基础版本（无LLM分析）**：
```bash
cd src
python main.py
```

**LLM分析版本（云端API）**：
```bash
cd src
python main_with_llm.py
```

**本地开源模型版本（免费，无需API key）**：
```bash
# 首次需要安装额外依赖
pip install transformers torch

cd src
python main_local.py
```

详细的本地模型使用说明请查看 [LOCAL_SETUP.md](LOCAL_SETUP.md)

### 使用示例

```
玩家1 ID (A-Z): B
玩家2 ID (A-Z): X
游戏回合数: 10
```

### 程序化调用

```python
from game import Game

# 模拟游戏：玩家B vs 玩家X，进行10回合
result = Game.simulate("B", "X", 10)

# 打印结果
print(result)

# 获取轨迹
player1_trajectory = result.get_trajectory_string(1)
player2_trajectory = result.get_trajectory_string(2)
```

## 项目结构

```
Markov/
├── src/
│   ├── core/                   # 核心游戏逻辑
│   │   ├── __init__.py
│   │   ├── players.py          # 玩家策略定义
│   │   └── game.py             # 游戏模拟逻辑
│   ├── analysis/               # 分析模块
│   │   ├── __init__.py
│   │   └── llm.py              # LLM分析模块
│   ├── main.py                 # 主程序入口（基础版）
│   └── main_with_llm.py        # LLM分析版主程序
├── test/
│   └── verify_distribution.py  # 分布验证测试
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

## 输出示例

```
============================================================
Game Result: B (Pure Rock) vs X (Win-Last)
Total Rounds: 10
============================================================

B (Pure Rock) trajectory:
Rock Rock Rock Rock Rock Rock Rock Rock Rock Rock

X (Win-Last) trajectory:
Scissors Paper Paper Paper Paper Paper Paper Paper Paper Paper

============================================================
Final Score:
  B (Pure Rock): 1 wins
  X (Win-Last): 9 wins
  Draws: 0
============================================================
```

## LLM分析功能

### 两种方式：

**方式1: 云端API（需要API key，付费）**
- 使用 `main_with_llm.py`
- 模型：qwen-plus, qwen-max等
- 优点：快速、质量最好
- 成本：每次分析约¥0.01-0.02

**方式2: 本地开源模型（完全免费）**
- 使用 `main_local.py`
- 模型：Qwen2.5-1.5B/3B/7B/14B等
- 优点：免费、隐私、无需联网
- 需求：6-16GB内存（CPU）或8-16GB显存（GPU）

详见：[云端API设置](SETUP.md) | [本地模型设置](LOCAL_SETUP.md)

### 配置LLM

**关于Qwen 3.5:**
- 如果Qwen 3.5已发布，使用 `qwen-max-latest` 获取最新版本
- 或直接尝试 `model_name="qwen3.5"`

**支持的模型:**
```python
# 推荐使用
model_name="qwen-plus"          # Qwen 2.5，平衡性能

# 其他选项  
model_name="qwen-turbo"         # 快速便宜
model_name="qwen-max-latest"    # 最新最强（可能是3.5）
model_name="qwen2.5-72b-instruct"  # 开源版本
```

查看完整模型列表和使用说明：[SETUP.md](SETUP.md)

## 技术说明

- **语言**: Python 3.6+
- **核心依赖**: requests (用于HTTP请求)
- **LLM**: 通义千问 Qwen API
- **设计模式**: 模块化设计，易于扩展

## 扩展

要添加新的玩家策略，在 `src/core/players.py` 的 `PLAYER_CONFIGS` 字典中添加新配置。
