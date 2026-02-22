# Parse Analysis Tool 使用说明

## 功能
解析LLM分析结果，从文本中提取结构化的JSON数据。

## 安装
无需额外安装，使用Python标准库。

## 使用方法

### 1. 基本用法 - 从文件解析
```bash
python tools/parse_analysis.py path/to/analysis.txt
```

### 2. 保存到JSON文件
```bash
python tools/parse_analysis.py path/to/analysis.txt -o output.json
```

### 3. 包含完整原始文本
```bash
python tools/parse_analysis.py path/to/analysis.txt -o output.json --full-text
```

### 4. 从stdin读取
```bash
cat analysis.txt | python tools/parse_analysis.py -
```

### 5. 批量处理
```bash
# PowerShell
Get-ChildItem src/analysis_results/*.txt | ForEach-Object {
    python tools/parse_analysis.py $_.FullName -o "$($_.BaseName)_parsed.json"
}

# Bash
for file in src/analysis_results/*.txt; do
    python tools/parse_analysis.py "$file" -o "${file%.txt}_parsed.json"
done
```

## 输出格式

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

## 期望的输入格式

LLM分析结果应包含 "Final Answer:" 部分，格式如下：

```
Final Answer:
Player1: X, 0.33, 0.33, 0.34
Player2: A, 0.0, 0.0, 1.0
```

## 示例

### 示例1: 解析单个文件
```bash
cd C:\Users\ASUS\Desktop\Markov
python tools/parse_analysis.py src/analysis_results/analysis_C_vs_D_20260223_004008.txt
```

### 示例2: 作为Python模块使用
```python
from tools.parse_analysis import parse_analysis_result

text = """
... (LLM分析文本)
Final Answer:
Player1: X, 0.33, 0.33, 0.34
Player2: A, 0.0, 0.0, 1.0
"""

result = parse_analysis_result(text)
print(result)
```

## 参数说明

- `input`: 输入文件路径（使用 `-` 表示从stdin读取）
- `-o, --output`: 输出JSON文件路径（默认输出到stdout）
- `--full-text`: 在JSON中包含完整的原始分析文本
- `--no-json`: 只显示解析状态，不输出JSON

## 错误处理

如果解析失败，工具会：
1. 返回 `parse_success: false`
2. 在 `error` 字段中说明失败原因
3. 在stderr输出警告信息
4. 退出码为1

## 注意事项

1. 确保LLM输出包含 "Final Answer:" 标记
2. 身份必须是 A-P 或 X-Z
3. 概率总和应接近1.0（允许5%误差）
