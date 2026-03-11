"""
验证 Union Loss 是否按照定义正确计算
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from evaluate_metrics import summarize

def test_union_loss_calculation():
    """
    测试 Union Loss 的计算是否符合定义：
    
    Union = 1/3 * (CE_norm + Brier_norm + EVLoss_norm)
    
    归一化规则：
    - CE_norm = (CE - min(CE)) / (max(CE) - min(CE))
    - Brier_norm = Brier (已在 [0,1])
    - EVLoss_norm = EVLoss / 4
    """
    print("=" * 70)
    print("测试 Union Loss 计算")
    print("=" * 70)
    
    # 模拟评估结果
    rows = [
        {
            "ACC": 1, "MDA": 1, "TV": 0.1, "WR_gap": 0.05,
            "CE": 1.0, "Brier": 0.1, "EVLoss": 0.2
        },
        {
            "ACC": 1, "MDA": 1, "TV": 0.2, "WR_gap": 0.08,
            "CE": 1.5, "Brier": 0.3, "EVLoss": 0.8
        },
        {
            "ACC": 0, "MDA": 1, "TV": 0.3, "WR_gap": 0.12,
            "CE": 2.0, "Brier": 0.5, "EVLoss": 1.6
        },
        {
            "ACC": 1, "MDA": 0, "TV": 0.4, "WR_gap": 0.15,
            "CE": 2.5, "Brier": 0.7, "EVLoss": 3.2
        },
        # 包含 Markov 玩家的样本（CE, Brier, EVLoss 为 None）
        {
            "ACC": 1, "MDA": 1, "TV": None, "WR_gap": 0.10,
            "CE": None, "Brier": None, "EVLoss": None
        },
    ]
    
    print("\n原始数据:")
    for i, row in enumerate(rows):
        if row["CE"] is not None:
            print(f"样本 {i+1}: CE={row['CE']:.1f}, Brier={row['Brier']:.1f}, EVLoss={row['EVLoss']:.1f}")
        else:
            print(f"样本 {i+1}: [Markov 玩家 - 无 CE/Brier/EVLoss]")
    
    # 调用 summarize 函数
    result = summarize(rows)
    
    print("\n" + "=" * 70)
    print("手动验证计算过程:")
    print("=" * 70)
    
    # 只考虑前4个样本（有完整指标的）
    ce_values = [1.0, 1.5, 2.0, 2.5]
    brier_values = [0.1, 0.3, 0.5, 0.7]
    evloss_values = [0.2, 0.8, 1.6, 3.2]
    
    print(f"\n有效样本: {len(ce_values)}")
    print(f"CE values: {ce_values}")
    print(f"Brier values: {brier_values}")
    print(f"EVLoss values: {evloss_values}")
    
    # CE 归一化
    ce_min = min(ce_values)
    ce_max = max(ce_values)
    ce_norm = [(c - ce_min) / (ce_max - ce_min) for c in ce_values]
    
    print(f"\n【CE 归一化】")
    print(f"  min = {ce_min}, max = {ce_max}")
    print(f"  CE_norm = {[f'{v:.3f}' for v in ce_norm]}")
    
    # Brier 归一化（不变）
    brier_norm = brier_values
    print(f"\n【Brier 归一化】")
    print(f"  Brier_norm = {[f'{v:.3f}' for v in brier_norm]}")
    print(f"  (已在 [0,1]，无需归一化)")
    
    # EVLoss 归一化
    evloss_norm = [e / 4.0 for e in evloss_values]
    print(f"\n【EVLoss 归一化】")
    print(f"  EVLoss_norm = EVLoss / 4")
    print(f"  EVLoss_norm = {[f'{v:.3f}' for v in evloss_norm]}")
    
    # 计算 Union Loss
    union_values = []
    print(f"\n【Union Loss 计算】")
    for i in range(len(ce_values)):
        union = (ce_norm[i] + brier_norm[i] + evloss_norm[i]) / 3.0
        union_values.append(union)
        print(f"  样本 {i+1}: ({ce_norm[i]:.3f} + {brier_norm[i]:.3f} + {evloss_norm[i]:.3f}) / 3 = {union:.3f}")
    
    union_mean = sum(union_values) / len(union_values)
    
    print(f"\n【最终结果】")
    print(f"  Union values: {[f'{v:.3f}' for v in union_values]}")
    print(f"  Union mean (手算): {union_mean:.6f}")
    print(f"  Union mean (代码): {result['Union']:.6f}")
    
    # 验证
    print("\n" + "=" * 70)
    if abs(result['Union'] - union_mean) < 1e-6:
        print("✅ 验证通过！Union Loss 计算正确")
    else:
        print("❌ 验证失败！Union Loss 计算有误")
        print(f"   期望值: {union_mean:.6f}")
        print(f"   实际值: {result['Union']:.6f}")
    print("=" * 70)
    
    # 显示完整结果
    print("\n完整评估结果:")
    for key, value in result.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: None")


if __name__ == "__main__":
    test_union_loss_calculation()
