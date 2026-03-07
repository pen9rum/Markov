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

## 🔧 工具集 (tools/)

### 1. 批量實驗工具 (batch_experiment.py)

運行批量游戲實驗和LLM分析

**使用方法**
```bash
python tools/batch_experiment.py
```

**功能**
- 自動排除雙Markov組合
- 兩種實驗類型：
  1. 都是非Markov（A-P vs A-P）
  2. 有一個Markov（X-Z vs A-P）
- 隨機抽取指定數量組合
- 使用相同LLM模型批量分析
- 結果保存為JSON格式

**輸出**
- 位置：`src/experiment_results/`
- 格式：`experiment_{type}_{timestamp}.json`
- 包含：真實身份、軌跡、LLM預測結果

---

### 2. 分析結果解析工具 (parse_analysis.py)

從LLM分析文本中提取結構化JSON數據，包含真實數據(ground truth)和LLM預測結果

**使用方法**

**基本用法（輸出到屏幕）：**
```bash
python tools/parse_analysis.py input.txt
```

**保存到JSON文件：**
```bash
python tools/parse_analysis.py input.txt -o output.json
```

**推薦的文件組織結構：**
```bash
# 按模型名稱組織輸出
python tools/parse_analysis.py "analysis_results/gemini-3-flash-preview/analysis.txt" -o "parsed_output/gemini-3-flash-preview/parsed_result.json"
```

**包含完整原始文本：**
```bash
python tools/parse_analysis.py input.txt -o output.json --full-text
```

**從stdin讀取：**
```bash
cat analysis.txt | python tools/parse_analysis.py -
```

**批量處理（PowerShell）：**
```powershell
# 確保目錄存在
mkdir -p parsed_output/gemini-3-flash-preview

# 批量解析
Get-ChildItem analysis_results/gemini-3-flash-preview/*.txt | ForEach-Object {
    $outputPath = "parsed_output/gemini-3-flash-preview/" + $_.BaseName + ".json"
    python tools/parse_analysis.py $_.FullName -o $outputPath
}
```

**輸出格式**
```json
{
  "parse_success": true,
  "ground_truth": {
    "player1_identity": "G",
    "player2_identity": "P",
    "player1": {
      "counts": {
        "rock": 0,
        "paper": 250,
        "scissors": 250
      },
      "probabilities": {
        "rock": 0.0,
        "paper": 0.5,
        "scissors": 0.5
      }
    },
    "player2": {
      "counts": {
        "rock": 84,
        "paper": 166,
        "scissors": 250
      },
      "probabilities": {
        "rock": 0.168,
        "paper": 0.332,
        "scissors": 0.5
      }
    }
  },
  "predictions": {
    "player1": {
      "identity": "G",
      "counts": {
        "rock": 0,
        "paper": 227,
        "scissors": 273
      },
      "probabilities": {
        "rock": 0.0,
        "paper": 0.454,
        "scissors": 0.546
      }
    },
    "player2": {
      "identity": "X",
      "counts": {
        "rock": 93,
        "paper": 134,
        "scissors": 273
      },
      "probabilities": {
        "rock": 0.186,
        "paper": 0.268,
        "scissors": 0.546
      }
    }
  },
  "markov_detection": null,
  "error": null
}
```

**支持的輸入格式**

LLM分析結果格式：
```
Final Answer:
Player1: G, 0.5, 0.3, 0.2
Player2: P, 0.167, 0.333, 0.5
```

或帶標籤的次數格式（推薦）：
```
**Final Answer:**
Player1: G, Rock 0, Paper 227, Scissors 273
Player2: P, Rock 17, Paper 33, Scissors 50
```

真實數據格式（從分析文件中自動提取）：
```
Match: G vs P

Player1 Actual Distribution:
  Rock: 0 (0.0%)
  Paper: 250 (50.0%)
  Scissors: 250 (50.0%)

Player2 Actual Distribution:
  Rock: 84 (16.8%)
  Paper: 166 (33.2%)
  Scissors: 250 (50.0%)
```

**參數說明**
- `input`: 輸入文件路徑（使用 `-` 表示從stdin讀取）
- `-o, --output`: 輸出JSON文件路徑（默認輸出到stdout）
- `--full-text`: 在JSON中包含完整的原始分析文本
- `--no-json`: 只顯示解析狀態，不輸出JSON

**作為Python模組使用**
```python
from tools.parse_analysis import parse_analysis_result

with open('analysis.txt', 'r', encoding='utf-8') as f:
    text = f.read()

result = parse_analysis_result(text)

# 檢查解析是否成功
if result['parse_success']:
    print(f"真實: {result['ground_truth']['player1_identity']} vs {result['ground_truth']['player2_identity']}")
    print(f"預測: {result['predictions']['player1']['identity']} vs {result['predictions']['player2']['identity']}")
```

**注意事項**
1. 確保LLM輸出包含 "Final Answer:" 或 "**Final Answer:**" 標記
2. 身份必須是 A-P 或 X-Z
3. 支持概率格式（0-1）和次數格式（自動轉換）
4. 自動從文件中提取真實的玩家身份和分布數據

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
├── tools/                         # 批量實驗和分析工具
│   ├── batch_experiment.py        # 批量實驗腳本
│   ├── parse_analysis.py          # 分析結果解析工具
│   └── README.md                  # 工具使用文檔
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

**相關文檔**
- [雲端API設置指南](SETUP.md)
- [本地模型設置指南](LOCAL_SETUP.md)
