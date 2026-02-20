# 本地开源模型使用指南

## 🎉 完全免费！无需API Key！

使用Hugging Face上的Qwen开源模型，完全本地运行，无需联网（首次下载后）。

## 📋 快速开始

### 1. 安装依赖

```bash
pip install transformers torch accelerate
```

**注意：**
- `transformers`: Hugging Face库
- `torch`: PyTorch深度学习框架
- `accelerate`: 加速模型加载（可选但推荐）

### 2. 运行程序

```bash
cd src
python main_local.py
```

**首次运行会自动下载模型**，根据模型大小需要5-30分钟。

## 🤖 推荐模型

### CPU可运行（无需显卡）

| 模型 | 参数量 | 内存需求 | 速度 | 推荐场景 |
|------|--------|----------|------|----------|
| Qwen2.5-1.5B-Instruct | 1.5B | ~3GB RAM | 很快 | 测试、简单分析 |
| Qwen2.5-3B-Instruct | 3B | ~6GB RAM | 快 | 日常使用 |

### GPU推荐（更好体验）

| 模型 | 参数量 | 显存需求 | 推荐显卡 |
|------|--------|----------|----------|
| Qwen2.5-7B-Instruct | 7B | ~14GB | RTX 3090/4090 |
| Qwen2.5-14B-Instruct | 14B | ~28GB | A100/H100 |
| Qwen2.5-32B-Instruct | 32B | ~64GB | 多卡/A100 |

## 💡 使用示例

### 基础使用
```bash
cd src
python main_local.py

# 程序会问：
# 1. 是否查看推荐模型？
# 2. 选择玩家和回合数
# 3. 是否进行LLM分析？
# 4. 选择模型（1.5B/3B/7B或自定义）
```

### 选择建议

**没有GPU？** → 选择 `1` (Qwen2.5-1.5B)
- CPU可跑，速度还行
- 适合测试和简单分析

**有中等GPU (8-16GB)?** → 选择 `3` (Qwen2.5-7B)
- 效果很好
- 推荐用这个

**只是想试试？** → 选择 `1` (Qwen2.5-1.5B)
- 最小最快
- 体验完整流程

## 🆚 云端API vs 本地模型

| 特性 | 云端API (main_with_llm.py) | 本地模型 (main_local.py) |
|------|----------------------------|--------------------------|
| **费用** | 需要付费（有免费额度） | 完全免费 |
| **API Key** | ✅ 必需 | ❌ 不需要 |
| **网络** | ✅ 必需 | ❌ 不需要（下载后） |
| **硬件要求** | 无 | CPU: 6-16GB RAM<br>GPU: 8-16GB VRAM |
| **速度** | 快 | 取决于硬件 |
| **模型版本** | 最新（可能有3.5） | 开源版本（2.5） |
| **质量** | qwen-max最好 | 7B/14B质量不错 |

## 📦 模型下载位置

默认下载到：`~/.cache/huggingface/hub/`

- Windows: `C:\Users\你的用户名\.cache\huggingface\hub\`
- Linux/Mac: `~/.cache/huggingface/hub/`

### 预下载模型（可选）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 只下载，不运行
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
```

## 🔧 高级配置

### 降低显存使用

```python
# 在 llm_local.py 中修改
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 半精度
    device_map="auto",
    load_in_8bit=True,  # 8位量化，进一步降低显存
    trust_remote_code=True
)
```

需要安装：`pip install bitsandbytes`

### 使用CPU

```python
# 强制使用CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
```

## ⚠️ 常见问题

### Q: Out of Memory错误？
A: 
1. 选择更小的模型（1.5B或3B）
2. 启用8位量化
3. 关闭其他程序释放内存

### Q: 下载很慢？
A: 
1. 使用镜像站：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 或者手动从镜像站下载

### Q: 分析质量不够好？
A: 
1. 使用更大的模型（7B或14B）
2. 调整temperature参数（降低到0.3-0.5）
3. 或者改用云端API的qwen-max

### Q: CPU运行太慢？
A: 
- Qwen2.5-1.5B在CPU上约30-60秒一次分析
- 3B模型会更慢（1-2分钟）
- 建议有GPU就用GPU

## 🎯 选择哪个版本？

**推荐云端API** 如果：
- ✅ 愿意付少量费用（每次分析几分钱）
- ✅ 需要最好的分析质量
- ✅ 没有强大的本地硬件

**推荐本地模型** 如果：
- ✅ 完全不想花钱
- ✅ 有8GB+内存或显卡
- ✅ 经常使用，想节省长期成本
- ✅ 关注隐私（数据不上传）

## 📚 参考资源

- [Qwen2.5模型仓库](https://huggingface.co/Qwen)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [模型下载镜像站](https://hf-mirror.com/)
