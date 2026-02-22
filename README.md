# 剪刀石头布模拟系统

一个支持多种策略的剪刀石头布（Rock-Paper-Scissors）游戏模拟系统，集成LLM分析功能。

## 📋 目录

- [功能特点](#-功能特点)
- [快速开始](#-快速开始)
- [玩家策略](#-玩家策略)
- [使用方法](#-使用方法)
- [LLM分析](#-llm分析)
- [批量实验工具](#-批量实验工具)
- [项目结构](#-项目结构)
- [输出示例](#-输出示例)
- [技术说明](#-技术说明)

---

## ✨ 功能特点

- **19种玩家策略**
  - 纯策略（总是出同一种）
  - 概率分布策略（按照给定概率随机出招）
  - 马可夫反应式策略（基于对手历史动作）

- **完整游戏追踪** - 记录每个回合双方的出招轨迹
- **详细统计分析** - 显示胜负平局的次数统计
- **LLM智能分析** - 使用大语言模型分析游戏策略和行为模式
- **批量实验工具** - 自动化批量实验和分析结果解析

---

## 🚀 快速开始

### 1. 环境设置

**克隆项目**
```bash
git clone <repository-url>
cd Markov
```

**激活虚拟环境**
```bash
# Windows PowerShell
.\markov_env\Scripts\Activate.ps1

# Windows CMD
.\markov_env\Scripts\activate.bat

# Linux/Mac
source markov_env/bin/activate
```

**安装依赖**
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥（可选 - 仅LLM云端分析需要）

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

获取API密钥：[阿里云百炼平台](https://dashscope.console.aliyun.com/)

### 3. 运行程序

**基础游戏模拟**
```bash
cd src
python main.py
```

**测试API连接**
```bash
python test_api.py
```

详细设置说明：[SETUP.md](SETUP.md) | [LOCAL_SETUP.md](LOCAL_SETUP.md)

---

## 🎮 玩家策略

### 纯策略 (Static)
| ID | 名称 | 说明 | 概率分布 (R/P/S) |
|----|------|------|------------------|
| **A** | Pure Scissors | 总是出剪刀 | 0 / 0 / 1 |
| **B** | Pure Rock | 总是出石头 | 1 / 0 / 0 |
| **C** | Pure Paper | 总是出布 | 0 / 1 / 0 |

### 概率分布策略 (Distribution)
| ID | 名称 | 概率分布 (R/P/S) |
|----|------|------------------|
| **D** | Uniform Random | 0.333 / 0.333 / 0.334 |
| **E** | Rock + Paper | 0.50 / 0.50 / 0 |
| **F** | Rock + Scissors | 0.50 / 0 / 0.50 |
| **G** | Paper + Scissors | 0 / 0.50 / 0.50 |
| **H** | Rock Biased | 0.50 / 0.25 / 0.25 |
| **I** | Paper Biased | 0.25 / 0.50 / 0.25 |
| **J** | Scissors Biased | 0.25 / 0.25 / 0.50 |
| **K** | Rock > Paper | 0.50 / 0.333 / 0.167 |
| **L** | Rock > Scissors | 0.50 / 0.167 / 0.333 |
| **M** | Paper > Rock | 0.333 / 0.50 / 0.167 |
| **N** | Paper > Scissors | 0.167 / 0.50 / 0.333 |
| **O** | Scissors > Rock | 0.333 / 0.167 / 0.50 |
| **P** | Scissors > Paper | 0.167 / 0.333 / 0.50 |

### 马可夫反应式策略 (Reactive/Markov)
| ID | 名称 | 说明 |
|----|------|------|
| **X** | Win-Last | 出能赢对手上一回合的手势 |
| **Y** | Lose-Last | 出会输给对手上一回合的手势 |
| **Z** | Copy-Last | 复制对手上一回合的手势 |

---

## 💻 使用方法

### 交互式运行

```bash
cd src
python main.py
```

**输入示例**
```
玩家1 ID (A-Z): B
玩家2 ID (A-Z): X
游戏回合数: 10
```

### 程序化调用

```python
from core.game import Game

# 模拟游戏：玩家B vs 玩家X，进行10回合
result = Game.simulate("B", "X", 10)

# 打印结果
print(result)

# 获取轨迹
player1_trajectory = result.get_trajectory_string(1)
player2_trajectory = result.get_trajectory_string(2)
```

### LLM分析模式

程序运行后会询问是否进行LLM分析：

```
是否使用LLM分析此局游戏？(y/n): y

选择LLM模型:
1. Qwen2.5-1.5B (本地，最快，CPU可跑)
2. Qwen2.5-3B (本地，平衡)
3. Qwen2.5-7B (本地，推荐，需要GPU)
4. 自定义本地模型名称
5. Qwen云端API (需要API key)

选择 (1-5): 
```

---

## 🤖 LLM分析

### 两种方式

#### 方式1: 云端API（推荐）
- **优点**: 快速、质量最好、无需本地资源
- **成本**: 约¥0.01-0.02/次分析
- **模型**: qwen-plus, qwen-max-latest等
- **要求**: API密钥

**支持的云端模型**
```python
model_name="qwen-plus"          # Qwen 2.5，平衡性能（推荐）
model_name="qwen-turbo"         # 快速便宜
model_name="qwen-max-latest"    # 最新最强
```

#### 方式2: 本地开源模型
- **优点**: 完全免费、隐私保护、无需联网
- **要求**: 6-16GB内存（CPU）或 8-16GB显存（GPU）
- **模型**: Qwen2.5-1.5B/3B/7B/14B等

**安装本地模型依赖**
```bash
pip install transformers torch
```

详见：[SETUP.md](SETUP.md) | [LOCAL_SETUP.md](LOCAL_SETUP.md)

---

## 🔧 批量实验工具

位于 `tools/` 目录，提供批量实验和分析结果解析功能。

### 1. 批量实验工具

**运行批量实验**
```bash
# Windows
tools\run_experiment.bat

# PowerShell
.\tools\run_experiment.ps1

# Linux/Mac
./tools/run_experiment.sh
```

**功能**
- 自动排除双马可夫组合
- 两种实验类型：
  - 都是非马可夫（A-P vs A-P）
  - 有一个马可夫（X-Z vs A-P）
- 随机抽取指定数量组合
- 批量LLM分析
- 结果保存为JSON格式

**输出位置**
```
src/experiment_results/
└── experiment_{type}_{timestamp}.json
```

### 2. 分析结果解析工具

从LLM分析文本中提取结构化JSON数据。

**基本用法**
```bash
# 输出到屏幕
python tools/parse_analysis.py input.txt

# 保存到JSON文件
python tools/parse_analysis.py input.txt -o output.json

# 使用快捷脚本（Windows）
tools\parse_analysis.bat input.txt output.json
```

**批量处理（PowerShell）**
```powershell
Get-ChildItem src/analysis_results/*.txt | ForEach-Object {
    python tools/parse_analysis.py $_.FullName -o "$($_.BaseName)_parsed.json"
}
```

**输出格式**
```json
{
  "parse_success": true,
  "players": {
    "player1": {
      "identity": "X",
      "probabilities": {"rock": 0.33, "paper": 0.33, "scissors": 0.34}
    },
    "player2": {
      "identity": "A",
      "probabilities": {"rock": 0.0, "paper": 0.0, "scissors": 1.0}
    }
  },
  "markov_detection": "player1"
}
```

详见：[tools/README.md](tools/README.md) | [tools/PARSE_ANALYSIS_README.md](tools/PARSE_ANALYSIS_README.md)

---

## 📁 项目结构

```
Markov/
├── src/
│   ├── core/                      # 核心游戏逻辑
│   │   ├── players.py             # 玩家策略定义
│   │   └── game.py                # 游戏模拟逻辑
│   ├── analysis/                  # 分析模块
│   │   ├── llm.py                 # LLM云端API分析
│   │   └── llm_local.py           # 本地模型分析
│   ├── main.py                    # 主程序入口
│   ├── analysis_results/          # 分析结果保存目录
│   └── experiment_results/        # 实验结果保存目录
├── tools/                         # 批量实验和分析工具
│   ├── batch_experiment.py        # 批量实验脚本
│   ├── parse_analysis.py          # 分析结果解析工具
│   ├── run_experiment.bat         # Windows批量实验启动脚本
│   ├── parse_analysis.bat         # Windows解析工具启动脚本
│   └── README.md                  # 工具使用文档
├── test/
│   └── verify_distribution.py     # 分布验证测试
├── markov_env/                    # Python虚拟环境
├── requirements.txt               # 项目依赖
├── README.md                      # 项目说明（本文件）
├── SETUP.md                       # 云端API设置说明
└── LOCAL_SETUP.md                 # 本地模型设置说明
```

---

## 📊 输出示例

### 游戏结果

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

### LLM分析结果

```
Final Answer:
Player1: B, 1.0, 0.0, 0.0
Player2: X, 0.0, 1.0, 0.0
```

---

## 🔬 技术说明

### 技术栈
- **语言**: Python 3.6+
- **核心依赖**: 
  - `requests` - HTTP请求（云端API）
  - `transformers` - 本地模型推理（可选）
  - `torch` - 深度学习框架（可选）
- **LLM**: 通义千问 Qwen API / 本地Qwen开源模型
- **设计模式**: 模块化设计，易于扩展

### 系统要求

**最低配置（基础游戏）**
- Python 3.6+
- 2GB RAM

**本地LLM分析配置**
- CPU: 8GB+ RAM（使用1.5B-3B模型）
- GPU: 8GB+ VRAM（使用7B+模型）

### 扩展开发

**添加新的玩家策略**
在 `src/core/players.py` 的 `PLAYER_CONFIGS` 字典中添加新配置：

```python
PLAYER_CONFIGS = {
    # ... 现有配置
    "NEW_ID": ("Strategy Name", rock_prob, paper_prob, scissors_prob, PlayerType.DISTRIBUTION, None),
}
```

**自定义LLM提示词**
编辑 `src/analysis/llm.py` 中的 `analyze_game_trajectory()` 函数。

---

## 📄 许可证

本项目遵循 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**相关文档**
- [云端API设置指南](SETUP.md)
- [本地模型设置指南](LOCAL_SETUP.md)
- [批量实验工具文档](tools/README.md)
- [分析解析工具文档](tools/PARSE_ANALYSIS_README.md)
