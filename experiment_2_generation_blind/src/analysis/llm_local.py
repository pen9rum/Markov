"""
本地LLM模块 - 使用Hugging Face Transformers运行开源模型
支持Qwen开源系列模型，无需API key
"""
from typing import Tuple, Dict, Any
import torch


def get_response_local(prompt: str,
                       model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                       max_length: int = 2048,
                       temperature: float = 0.7,
                       **kwargs) -> Tuple[dict, str]:
    """
    使用本地Hugging Face模型获取响应
    
    Args:
        prompt: 输入提示词
        model_name: 模型名称，支持：
                   - "Qwen/Qwen2.5-7B-Instruct" (推荐，7B参数，约14GB显存)
                   - "Qwen/Qwen2.5-14B-Instruct" (14B参数，约28GB显存)
                   - "Qwen/Qwen2.5-3B-Instruct" (3B参数，约6GB显存，轻量级)
                   - "Qwen/Qwen2.5-1.5B-Instruct" (1.5B参数，约3GB显存，最轻)
        max_length: 最大生成长度
        temperature: 温度参数 (0-1)
        **kwargs: 其他生成参数
    
    Returns:
        (元数据字典, 输出文本)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "需要安装transformers库。请运行: pip install transformers torch"
        )
    
    print(f"加载模型: {model_name}")
    print("首次运行会下载模型，可能需要几分钟...")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度节省显存
        device_map="auto",  # 自动选择设备(GPU/CPU)
        trust_remote_code=True
    )
    
    # 构建消息
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用chat模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    print("生成中...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        **kwargs
    )
    
    # 解码
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 构建响应
    metadata = {
        "model": model_name,
        "device": str(model.device),
        "input_tokens": len(model_inputs.input_ids[0]),
        "output_tokens": len(generated_ids[0]),
        "total_tokens": len(model_inputs.input_ids[0]) + len(generated_ids[0])
    }
    
    return metadata, response_text


def list_recommended_models():
    """列出推荐的开源模型"""
    models = {
        "轻量级 (CPU可跑)": [
            ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B参数, ~3GB RAM, 最快"),
            ("Qwen/Qwen2.5-3B-Instruct", "3B参数, ~6GB RAM, 平衡"),
        ],
        "标准 (需要GPU)": [
            ("Qwen/Qwen2.5-7B-Instruct", "7B参数, ~14GB VRAM, 推荐"),
            ("Qwen/Qwen2.5-14B-Instruct", "14B参数, ~28GB VRAM, 强大"),
        ],
        "高级 (需要大显存)": [
            ("Qwen/Qwen2.5-32B-Instruct", "32B参数, ~64GB VRAM, 很强"),
            ("Qwen/Qwen2.5-72B-Instruct", "72B参数, ~144GB VRAM, 最强"),
        ]
    }
    
    print("\n推荐的开源Qwen模型:")
    print("="*70)
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model_name, description in model_list:
            print(f"  • {model_name}")
            print(f"    {description}")
    print("\n" + "="*70)
    
    return models


if __name__ == "__main__":
    # 测试本地模型
    print("测试本地Qwen模型...")
    list_recommended_models()
    
    print("\n尝试使用最轻量级模型进行测试...")
    try:
        metadata, text = get_response_local(
            "你好，请介绍一下你自己。",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_length=100
        )
        print(f"\n成功! 模型响应:")
        print(text)
        print(f"\nToken使用: {metadata}")
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示: 确保已安装 transformers 和 torch")
        print("运行: pip install transformers torch")
