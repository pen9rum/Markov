"""
测试Qwen API连接
"""
import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.llm import get_response


def test_api_connection():
    """测试API连接"""
    print("\n" + "="*60)
    print("Qwen API 连接测试")
    print("="*60 + "\n")
    
    # 检查环境变量
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("\n请先设置API密钥:")
        print("  Windows: set DASHSCOPE_API_KEY=your_key_here")
        print("  Linux/Mac: export DASHSCOPE_API_KEY=your_key_here")
        return
    
    print(f"✓ API密钥已设置: {api_key[:8]}...{api_key[-4:]}")
    
    # 测试不同模型
    models_to_test = [
        ("qwen-plus", "推荐使用，平衡性能"),
        ("qwen-turbo", "快速响应"),
        ("qwen-max-latest", "最新版本（可能包括3.5）"),
    ]
    
    for model_name, description in models_to_test:
        print(f"\n测试模型: {model_name}")
        print(f"说明: {description}")
        print("-" * 60)
        
        try:
            response, text = get_response(
                prompt="请用一句话介绍你自己，包括你的版本号。",
                model_name=model_name
            )
            
            print(f"✓ 成功!")
            print(f"  模型: {response.get('model')}")
            print(f"  响应: {text[:100]}...")
            
            usage = response.get('usage', {})
            if usage:
                print(f"  Token使用: 输入={usage.get('prompt_tokens')} "
                      f"输出={usage.get('completion_tokens')} "
                      f"总计={usage.get('total_tokens')}")
        
        except Exception as e:
            print(f"❌ 失败: {e}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_api_connection()
