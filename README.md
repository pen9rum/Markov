# 剪刀石头布模拟系统

一个支持多种策略的剪刀石头布（Rock-Paper-Scissors）游戏模拟系统，集成LLM分析功能和完整的实验工具链。

## 📋 目录

- [功能特点](#-功能特点)
- [快速开始](#-快速开始)
- [玩家策略](#-玩家策略)
- [工具链与 Pipeline](#-工具链与-pipeline)
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
- **完整工具链** - 从实验执行到结果评估的端到端自动化流程

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

创建 `.env` 文件（會自動載入）：
```bash
# Qwen API
DASHSCOPE_API_KEY=your_qwen_api_key

# Gemini API
GEMINI_API_KEY=your_gemini_api_key

# OpenAI API (GPT-5-mini)
OPENAI_API_KEY=your_openai_api_key

# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key
```

或使用環境變量：
```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"
$env:GEMINI_API_KEY="your_gemini_key"

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
export GEMINI_API_KEY=your_gemini_key
```

獲取API密鑰：
- Qwen: [阿里云百炼平台](https://dashscope.console.aliyun.com/)
- Gemini: [Google AI Studio](https://aistudio.google.com/apikey)
- OpenAI: [OpenAI Platform](https://platform.openai.com/)
- DeepSeek: [DeepSeek Platform](https://platform.deepseek.com/)

### 3. 运行程序

**基础游戏模拟**
```bash
python src/main.py
```

**批量实验（推荐）**
```bash
# 運行 10 組實驗，每組 100 回合，使用 GPT-5-mini
python tools/batch_experiment.py --type1 10 --type2 10 --rounds 100 --model gpt-5-mini

# 運行所有可能組合（type1: 240組, type2: 78組）
python tools/batch_experiment.py --all --rounds 100 --model deepseek-chat
```

详细设置说明：[SETUP.md](SETUP.md) | [LOCAL_SETUP.md](LOCAL_SETUP.md)

---

## 🔧 工具链与 Pipeline

### 完整實驗流程

```
┌─────────────────────────────────────────────────────────────┐
│                    實驗執行階段                              │
├─────────────────────────────────────────────────────────────┤
│ 1️⃣ batch_experiment.py                                      │
│    └─ 批量執行實驗，生成 LLM 分析文本                       │
│       輸出: batch_results/{model}/{rounds}/type1|type2/*.txt │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    解析階段                                  │
├─────────────────────────────────────────────────────────────┤
│ 2️⃣ batch_parser.py                                          │
│    └─ 解析 LLM 輸出為結構化 JSON                            │
│       輸出: parsed_output/{model}/{rounds}/type1|type2/*.json│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    評估階段                                  │
├─────────────────────────────────────────────────────────────┤
│ 3️⃣ evaluate_metrics.py                                      │
│    └─ 計算準確率、TV距離、勝率差距等指標                    │
│       輸出: evaluation_summary.json, evaluation_detail.json  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    匯出階段                                  │
├─────────────────────────────────────────────────────────────┤
│ 4️⃣ export_metrics_csv.py                                    │
│    └─ 匯出多模型、多 rounds 結果為 CSV                      │
│       輸出: metrics_export.csv                               │
└─────────────────────────────────────────────────────────────┘
```

### 工具詳細說明

#### 1. 實驗工具

| 工具 | 說明 | 用途 |
|------|------|------|
| **`src/main.py`** | 單次實驗執行器 | 互動式執行單場遊戲，支援 9 種 LLM 模型 |
| **`tools/batch_experiment.py`** | 批量實驗執行器 | 自動執行大量實驗組合，支援隨機抽樣或全組合 |

**批量實驗使用範例：**
```bash
# 隨機抽樣模式
python tools/batch_experiment.py --type1 10 --type2 5 --rounds 100 --model gpt-5-mini

# 全組合模式（240+78=318組）
python tools/batch_experiment.py --all --rounds 200 --model deepseek-chat

# 支援的模型
--model qwen-1.5b | qwen-3b | qwen-7b | qwen-api | gemini | 
        gpt-5-mini | deepseek-chat | deepseek-reasoner
```

**組合統計：**
- **Type1（非Markov vs 非Markov）**: 240 組
  - A-P (16個玩家) vs A-P（排除自己對戰）= 16 × 15 = 240
- **Type2（非Markov vs Markov）**: 78 組
  - (D-P，排除A,B,C) vs (X,Y,Z)，雙向 = 13 × 3 × 2 = 78
- **總計**: 318 組

#### 2. 解析工具

| 工具 | 說明 | 用途 |
|------|------|------|
| **`tools/batch_parser.py`** | 批量解析器 | 將 LLM 文本輸出解析為結構化 JSON |
| **`tools/parse_analysis.py`** | 單文件解析器 | 解析單個分析文件（被 batch_parser 調用）|

**批量解析使用範例：**
```bash
# 解析特定模型的特定 rounds
python tools/batch_parser.py --model deepseek-chat --rounds 200

# 解析特定模型的所有 rounds
python tools/batch_parser.py --model gpt-5-mini

# 解析所有模型的所有 rounds
python tools/batch_parser.py
```

#### 3. 評估工具

| 工具 | 說明 | 用途 |
|------|------|------|
| **`tools/evaluate_metrics.py`** | 指標評估器 | 計算 ACC、MDA、TV、WR_gap 等指標 |
| **`tools/export_metrics_csv.py`** | CSV 匯出器 | 匯出多模型多 rounds 的結果為 CSV |

**評估使用範例：**
```bash
# 評估單個模型的單個 rounds
python tools/evaluate_metrics.py --folder "parsed_output/gpt-5-mini/200"

# 批量匯出多個模型多個 rounds
python tools/export_metrics_csv.py --models gpt-5-mini deepseek-chat --rounds 100 200 500 1000 --output results.csv
```

**評估指標說明：**

**基礎指標：**
- **ACC (Accuracy)**: 玩家身份識別準確率（兩個玩家都對才算對）
- **MDA (Markov Detection Accuracy)**: Markov 玩家檢測準確率
- **TV (Total Variation)**: 預測分布與真實分布的 TV 距離（越小越好）
- **WR_gap (Win Rate Gap)**: 預測勝率與真實勝率的差距（越小越好）

**進階指標（新增）：**
- **CE (Cross Entropy)**: 交叉熵，衡量預測分布編碼真實分布的信息損失
- **Brier Score**: L2距離，同時懲罰排序和校準誤差
- **EVLoss (Expected Value Loss)**: 期望值損失，關注勝率優勢的差異
- **Union Loss**: CE、Brier、EVLoss 的歸一化平均值（綜合評估指標）

詳細說明請參閱：[評估指標詳細說明](#-評估指標詳細說明)

#### 4. 驗證工具

| 工具 | 說明 | 用途 |
|------|------|------|
| **`test/verify_distribution.py`** | 分布驗證器 | 驗證玩家策略的統計分布是否正確 |

### 目錄結構

```
Markov/
├── src/                        # 核心源碼
│   ├── main.py                 # 主程序入口
│   ├── core/                   # 遊戲核心邏輯
│   │   ├── game.py             # 遊戲模擬引擎
│   │   └── players.py          # 玩家策略定義
│   └── analysis/               # LLM 分析模組
│       ├── llm.py              # 雲端 API (Qwen/Gemini/OpenAI/DeepSeek)
│       └── llm_local.py        # 本地模型
│
├── tools/                      # 工具集
│   ├── batch_experiment.py     # 批量實驗執行器
│   ├── batch_parser.py         # 批量解析器
│   ├── parse_analysis.py       # 解析邏輯
│   ├── evaluate_metrics.py     # 指標評估器
│   └── export_metrics_csv.py   # CSV 匯出器
│
├── batch_results/              # 實驗結果（被 gitignore）
│   └── {model}/
│       └── {rounds}/
│           ├── type1_non_markov/
│           └── type2_with_markov/
│
├── parsed_output/              # 解析結果（被 gitignore）
│   └── {model}/
│       └── {rounds}/
│           ├── type1_non_markov/
│           ├── type2_with_markov/
│           ├── evaluation_summary.json
│           └── evaluation_detail.json
│
└── test/                       # 測試工具
    └── verify_distribution.py  # 分布驗證
```

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

## � 評估指標詳細說明

### 基礎指標

#### ACC (Accuracy) - 玩家識別準確率
- **定義**: 雙方玩家身份都預測正確才算正確
- **範圍**: [0, 1]
- **越大越好**

#### MDA (Markov Detection Accuracy) - Markov 檢測準確率
- **定義**: 正確判斷遊戲中是否包含 Markov 玩家
- **範圍**: [0, 1]
- **越大越好**

#### TV (Total Variation) - 分布總變差距離
- **定義**: `TV(q, p) = 1/2 * Σ|q_i - p_i|`
- **範圍**: [0, 1]
- **越小越好**

#### WR_gap (Win Rate Gap) - 勝率差距
- **定義**: `|WR_真實 - WR_預測|`
- **範圍**: [0, 1]
- **越小越好**

### 進階指標

#### 1️⃣ Cross Entropy (CE) - 交叉熵

**定義：**
```
CE(p*, p^) = -Σ p*_c * log(p^_c + ε)
其中 c ∈ {win, draw, loss}
```

**含義：**
- 用真實分布 p* 去衡量用模型分布 p^ 編碼時產生的信息驚訝度
- 如果模型把高概率放在真正會發生的事件上 → CE 會小

**特點：**
- 不一定等於 0（除非真實分布是 deterministic 且預測完全正確）
- 取值範圍：[0, ∞)（實際使用時會做歸一化）
- **越小越好**

#### 2️⃣ Brier Score - 布賴爾分數

**定義：**
```
Brier(p*, p^) = Σ (p^_c - p*_c)²
```

**含義：**
- 本質是 L2 距離（歐氏距離的平方）
- 同時懲罰排序錯誤 (ranking error) 和校準錯誤 (calibration error)

**特點：**
- 取值範圍：[0, 1]
- 完全相同時 = 0
- **越小越好**

#### 3️⃣ Expected Value Loss (EVLoss) - 期望值損失

**定義：**
```
EV(p) = p_win - p_loss
EVLoss = (EV(p*) - EV(p^))²
```

**含義：**
- 衡量模型對勝率優勢估計錯多少
- 關注的是結果的期望值差異，而不是分布本身

**特點：**
- EV(p) ∈ [-1, 1]
- EVLoss ∈ [0, 4]
- **越小越好**

#### 4️⃣ Union Loss - 綜合損失

**定義：**
```
Union = 1/3 * (CE_norm + Brier_norm + EVLoss_norm)
```

**歸一化方法：**
- **CE**: min-max 歸一化 → `(CE - min) / (max - min)`
- **Brier**: 已在 [0,1]，不需要歸一化
- **EVLoss**: 除以最大值 4 → `EVLoss / 4`

**特點：**
- 綜合了三個不同角度的誤差
- 取值範圍：[0, 1]
- **越小越好**

### 指標對比總結

| 指標 | 衡量內容 | 範圍 | 方向 |
|------|---------|------|------|
| **ACC** | 玩家識別準確率 | [0, 1] | 越大越好 |
| **MDA** | Markov檢測準確率 | [0, 1] | 越大越好 |
| **TV** | 分布總變差距離 | [0, 1] | 越小越好 |
| **WR_gap** | 勝率差距 | [0, 1] | 越小越好 |
| **CE** | 交叉熵 | [0, ∞) | 越小越好 |
| **Brier** | 布賴爾分數 | [0, 1] | 越小越好 |
| **EVLoss** | 期望值損失 | [0, 4] | 越小越好 |
| **Union** | 綜合損失 | [0, 1] | 越小越好 |

### 推薦使用指南

- **整體性能評估**: Union Loss
- **分布準確性**: TV 和 Brier
- **結果準確性**: WR_gap 和 EVLoss
- **識別能力**: ACC 和 MDA

**注意事項：**
1. CE、Brier、EVLoss 只對非 Markov 玩家計算（因為 Markov 玩家的分布依賴於對手）
2. Union Loss 的歸一化是動態的，不同批次的值可能因歸一化範圍不同而不可直接比較

---

## 🔧 詳細設置指南

### 雲端 API 設置（Qwen/Gemini/OpenAI/DeepSeek）

#### 1. 獲取 API 密鑰

**Qwen API（阿里雲百煉）：**
1. 訪問 [阿里云百炼平台](https://dashscope.console.aliyun.com/)
2. 登錄/註冊阿里云賬號
3. 進入「API-KEY管理」頁面
4. 創建新的 API-KEY 並複製

**Gemini API（Google）：**
1. 訪問 [Google AI Studio](https://aistudio.google.com/apikey)
2. 登錄 Google 賬號
3. 創建 API 密鑰

**OpenAI API：**
1. 訪問 [OpenAI Platform](https://platform.openai.com/)
2. 創建帳號並設置付費
3. 生成 API 密鑰

**DeepSeek API：**
1. 訪問 [DeepSeek Platform](https://platform.deepseek.com/)
2. 註冊並獲取 API 密鑰

#### 2. 設置環境變量

**Windows (PowerShell):**
```powershell
$env:DASHSCOPE_API_KEY="your_qwen_api_key"
$env:GEMINI_API_KEY="your_gemini_api_key"
$env:OPENAI_API_KEY="your_openai_api_key"
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
```

**Windows (CMD):**
```cmd
set DASHSCOPE_API_KEY=your_api_key_here
set GEMINI_API_KEY=your_gemini_api_key
```

**Linux/Mac:**
```bash
export DASHSCOPE_API_KEY=your_api_key_here
export GEMINI_API_KEY=your_gemini_api_key
```

**永久設置 (Windows):**
```powershell
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'your_api_key_here', 'User')
```

**或使用 .env 文件（推薦）：**
在項目根目錄創建 `.env` 文件：
```bash
DASHSCOPE_API_KEY=your_qwen_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 本地模型設置

#### 優點
- ✅ 完全免費
- ✅ 隱私保護
- ✅ 無需聯網（首次下載後）

#### 系統要求

**CPU 運行（無需顯卡）：**
| 模型 | 參數量 | 記憶體需求 | 速度 | 推薦場景 |
|------|--------|----------|------|---------|
| Qwen2.5-1.5B-Instruct | 1.5B | ~3GB RAM | 很快 | 測試、簡單分析 |
| Qwen2.5-3B-Instruct | 3B | ~6GB RAM | 快 | 日常使用 |

**GPU 運行（更好體驗）：**
| 模型 | 參數量 | 顯存需求 | 推薦顯卡 |
|------|--------|----------|----------|
| Qwen2.5-7B-Instruct | 7B | ~14GB | RTX 3090/4090 |
| Qwen2.5-14B-Instruct | 14B | ~28GB | A100/H100 |

#### 安裝步驟

1. **安裝依賴**
```bash
pip install transformers torch accelerate
```

2. **運行程序**
```bash
cd src
python main.py
```

3. **選擇本地模型**
- 首次運行會自動下載模型（根據模型大小需要 5-30 分鐘）
- 下載完成後會緩存，以後無需重新下載

#### 模型選擇建議

```python
# 快速測試（CPU 可跑）
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 日常使用（CPU 可跑）
model_name = "Qwen/Qwen2.5-3B-Instruct"

# 最佳效果（需要 GPU）
model_name = "Qwen/Qwen2.5-7B-Instruct"
```

---

## �🔬 技术说明

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

