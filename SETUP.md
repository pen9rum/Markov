# Qwen API 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install requests
```

或者使用requirements.txt：
```bash
pip install -r requirements.txt
```

### 2. 获取API密钥

1. 访问 [阿里云百炼平台](https://dashscope.console.aliyun.com/)
2. 登录/注册阿里云账号
3. 进入"API-KEY管理"页面
4. 创建新的API-KEY并复制

### 3. 设置环境变量

**Windows (PowerShell):**
```powershell
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

**Windows (CMD):**
```cmd
set DASHSCOPE_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

**永久设置 (Windows):**
```powershell
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'your_api_key_here', 'User')
```

### 4. 运行程序

```bash
cd src
python main_with_llm.py
```

## 支持的模型

| 模型名称 | 版本 | 特点 | 推荐场景 |
|---------|------|------|---------|
| `qwen-turbo` | Qwen 2.5 | 快速、便宜 | 简单分析 |
| `qwen-plus` | Qwen 2.5 | 平衡性能 | **推荐使用** |
| `qwen-max` | 最新 | 最强性能 | 深度分析 |
| `qwen-max-latest` | 最新 | 可能包括3.5 | 最新功能 |
| `qwen2.5-72b-instruct` | 2.5 | 开源版本 | 离线部署 |
| `qwen2.5-32b-instruct` | 2.5 | 中等规模 | 本地测试 |
| `qwen2.5-14b-instruct` | 2.5 | 小规模 | 轻量级 |
| `qwen2.5-7b-instruct` | 2.5 | 最小 | 资源受限 |

## 关于Qwen 3.5

**目前状态：**
- 截至2026年2月，阿里云API主要提供Qwen 2.5系列
- 如果Qwen 3.5已发布，可通过 `qwen-max-latest` 访问最新版本
- 或直接指定 `qwen3.5` (如果API已支持)

**检查可用模型：**
```python
# 在代码中尝试不同模型名称
model_name = "qwen-max-latest"  # 或 "qwen3.5"
```

## 使用示例

### 基础使用
```bash
cd src
python main_with_llm.py

# 输入：
# 玩家1 ID: X
# 玩家2 ID: B
# 游戏回合数: 100
# 是否使用LLM分析: y
```

### 更改模型
在 `main_with_llm.py` 中修改：
```python
analysis_result = analyze_game_trajectory(
    ...,
    model_name="qwen-max-latest"  # 改为你想要的模型
)
```

### 自定义参数
```python
from analysis.llm import get_response

response, text = get_response(
    prompt="你的问题",
    model_name="qwen-plus",
    temperature=0.7,      # 创造性 (0-1)
    max_tokens=2000,      # 最大输出长度
    top_p=0.9            # 采样参数
)
```

## 常见问题

### Q: 如何知道我能用哪个版本？
A: 访问 [阿里云模型广场](https://help.aliyun.com/zh/model-studio/getting-started/models) 查看最新支持的模型列表。

### Q: API调用失败怎么办？
A: 检查：
1. API密钥是否正确设置
2. 账户余额是否充足
3. 模型名称是否正确
4. 网络连接是否正常

### Q: 如何测试API是否可用？
A: 运行测试脚本：
```python
from analysis.llm import get_response

try:
    response, text = get_response("Hello", model_name="qwen-plus")
    print("API可用！")
    print(f"模型: {response.get('model')}")
    print(f"响应: {text}")
except Exception as e:
    print(f"错误: {e}")
```

### Q: 费用如何计算？
A: 
- qwen-turbo: ~¥0.002/1k tokens
- qwen-plus: ~¥0.004/1k tokens  
- qwen-max: ~¥0.04/1k tokens
- 详见阿里云官网最新定价

## 完整工作流程

```
1. 安装依赖
   ↓
2. 获取API密钥
   ↓
3. 设置环境变量
   ↓
4. 运行 main_with_llm.py
   ↓
5. 输入玩家和回合数
   ↓
6. 游戏模拟完成
   ↓
7. 选择 'y' 进行LLM分析
   ↓
8. 等待AI分析结果
   ↓
9. 查看分析报告
   ↓
10. 可选：保存到文件
```

## 参考链接

- [阿里云百炼平台](https://dashscope.console.aliyun.com/)
- [Qwen API文档](https://help.aliyun.com/zh/dashscope/)
- [模型列表](https://help.aliyun.com/zh/model-studio/getting-started/models)
- [定价说明](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-pricing)
