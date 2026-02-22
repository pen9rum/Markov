# Markov Tools

批量实验和分析工具集

## 工具列表

### 1. 批量实验工具 (batch_experiment.py)
运行批量游戏实验和LLM分析

#### Windows
```bash
# 双击运行
run_experiment.bat

# 或使用PowerShell
.\run_experiment.ps1
```

#### Linux/Mac
```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

#### 功能
- 自动排除双Markov组合
- 两种实验类型：
  1. 都是非Markov（A-P vs A-P）
  2. 有一个Markov（X-Z vs A-P）
- 随机抽取指定数量组合
- 使用相同LLM模型批量分析
- 结果保存为JSON格式

#### 输出
- 位置：`src/experiment_results/`
- 格式：`experiment_{type}_{timestamp}.json`
- 包含：真实身份、轨迹、LLM预测结果

---

### 2. 分析结果解析工具 (parse_analysis.py)
从LLM分析文本中提取结构化JSON数据

#### 使用方法

**基本用法（输出到屏幕）：**
```bash
python tools/parse_analysis.py input.txt
```

**保存到JSON文件：**
```bash
python tools/parse_analysis.py input.txt -o output.json

# 或使用快捷脚本（Windows）
tools\parse_analysis.bat input.txt output.json
```

**包含完整原始文本：**
```bash
python tools/parse_analysis.py input.txt -o output.json --full-text
```

**从stdin读取：**
```bash
cat analysis.txt | python tools/parse_analysis.py -
```

**批量处理（PowerShell）：**
```powershell
Get-ChildItem src/analysis_results/*.txt | ForEach-Object {
    python tools/parse_analysis.py $_.FullName -o "$($_.BaseName)_parsed.json"
}
```

#### 输出格式
```json
{
  "parse_success": true,
  "players": {
    "player1": {
      "identity": "X",
      "probabilities": {
        "rock": 0.33,
        "paper": 0.33,
        "scissors": 0.34
      }
    },
    "player2": {
      "identity": "A",
      "probabilities": {
        "rock": 0.0,
        "paper": 0.0,
        "scissors": 1.0
      }
    }
  },
  "markov_detection": "player1",
  "error": null
}
```

#### 期望输入格式
LLM分析结果应包含 "Final Answer:" 部分：
```
Final Answer:
Player1: X, 0.33, 0.33, 0.34
Player2: A, 0.0, 0.0, 1.0
```

详细文档请查看：[PARSE_ANALYSIS_README.md](PARSE_ANALYSIS_README.md)

---

## 快捷脚本说明

### run_experiment.bat / run_experiment.ps1 / run_experiment.sh
- 启动批量实验工具
- 交互式选择实验参数
- 自动运行游戏模拟和LLM分析

### parse_analysis.bat
- 快速解析单个分析文件
- 用法：`parse_analysis.bat input.txt [output.json]`
- 如果不指定output，则输出到屏幕
