"""
测试新添加的指标: CE, Brier, EVLoss, Union
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from evaluate_metrics import cross_entropy, brier_score, ev_loss
import math

def test_cross_entropy():
    """测试 Cross Entropy"""
    print("=" * 60)
    print("测试 Cross Entropy (CE)")
    print("=" * 60)
    
    # Case 1: 完全相同的分布
    p_star = (0.5, 0.3, 0.2)
    p_hat = (0.5, 0.3, 0.2)
    ce = cross_entropy(p_star, p_hat)
    print(f"\nCase 1: 相同分布")
    print(f"p* = {p_star}")
    print(f"p^ = {p_hat}")
    print(f"CE = {ce:.6f}")
    print(f"理论值: entropy(p*) ≈ {-sum(p*math.log(p) for p in p_star):.6f}")
    
    # Case 2: 完全不同的分布
    p_star = (0.7, 0.2, 0.1)
    p_hat = (0.1, 0.2, 0.7)
    ce = cross_entropy(p_star, p_hat)
    print(f"\nCase 2: 不同分布")
    print(f"p* = {p_star}")
    print(f"p^ = {p_hat}")
    print(f"CE = {ce:.6f}")
    
    # Case 3: Deterministic (p* = (1,0,0))
    p_star = (1.0, 0.0, 0.0)
    p_hat = (1.0, 0.0, 0.0)
    ce = cross_entropy(p_star, p_hat)
    print(f"\nCase 3: Deterministic 且相同")
    print(f"p* = {p_star}")
    print(f"p^ = {p_hat}")
    print(f"CE = {ce:.6f}")
    print(f"应该接近 0")


def test_brier_score():
    """测试 Brier Score"""
    print("\n" + "=" * 60)
    print("测试 Brier Score")
    print("=" * 60)
    
    # Case 1: 完全相同
    p_star = (0.5, 0.3, 0.2)
    p_hat = (0.5, 0.3, 0.2)
    brier = brier_score(p_star, p_hat)
    print(f"\nCase 1: 相同分布")
    print(f"p* = {p_star}")
    print(f"p^ = {p_hat}")
    print(f"Brier = {brier:.6f}")
    print(f"应该 = 0")
    
    # Case 2: 例子
    p_star = (0.5, 0.3, 0.2)
    p_hat = (0.6, 0.2, 0.2)
    brier = brier_score(p_star, p_hat)
    expected = (0.6-0.5)**2 + (0.2-0.3)**2 + (0.2-0.2)**2
    print(f"\nCase 2: 不同分布")
    print(f"p* = {p_star}")
    print(f"p^ = {p_hat}")
    print(f"Brier = {brier:.6f}")
    print(f"手算 = {expected:.6f}")
    
    # Case 3: 完全相反
    p_star = (0.9, 0.05, 0.05)
    p_hat = (0.05, 0.05, 0.9)
    brier = brier_score(p_star, p_hat)
    print(f"\nCase 3: 完全相反")
    print(f"p* = {p_star}")
    print(f"p^ = {p_hat}")
    print(f"Brier = {brier:.6f}")
    print(f"应该很大 (接近最大值)")


def test_ev_loss():
    """测试 EV Loss"""
    print("\n" + "=" * 60)
    print("测试 EV Loss")
    print("=" * 60)
    
    # Case 1: 相同分布
    p_star = (0.5, 0.3, 0.2)
    p_hat = (0.5, 0.3, 0.2)
    evloss = ev_loss(p_star, p_hat)
    print(f"\nCase 1: 相同分布")
    print(f"p* = {p_star}, EV* = {p_star[0] - p_star[2]:.3f}")
    print(f"p^ = {p_hat}, EV^ = {p_hat[0] - p_hat[2]:.3f}")
    print(f"EVLoss = {evloss:.6f}")
    print(f"应该 = 0")
    
    # Case 2: 例子
    p_star = (0.6, 0.2, 0.2)
    p_hat = (0.4, 0.2, 0.4)
    ev_star = 0.6 - 0.2
    ev_hat = 0.4 - 0.4
    expected = (ev_star - ev_hat) ** 2
    evloss = ev_loss(p_star, p_hat)
    print(f"\nCase 2: 不同 EV")
    print(f"p* = {p_star}, EV* = {ev_star:.3f}")
    print(f"p^ = {p_hat}, EV^ = {ev_hat:.3f}")
    print(f"EVLoss = {evloss:.6f}")
    print(f"手算 = {expected:.6f}")
    
    # Case 3: 极端情况
    p_star = (1.0, 0.0, 0.0)
    p_hat = (0.0, 0.0, 1.0)
    ev_star = 1.0 - 0.0
    ev_hat = 0.0 - 1.0
    expected = (ev_star - ev_hat) ** 2
    evloss = ev_loss(p_star, p_hat)
    print(f"\nCase 3: 极端相反")
    print(f"p* = {p_star}, EV* = {ev_star:.3f}")
    print(f"p^ = {p_hat}, EV^ = {ev_hat:.3f}")
    print(f"EVLoss = {evloss:.6f}")
    print(f"手算 = {expected:.6f}")
    print(f"最大值 = 4 (当 EV* = 1, EV^ = -1 或相反)")


def test_union_logic():
    """测试 Union Loss 的逻辑"""
    print("\n" + "=" * 60)
    print("测试 Union Loss 归一化逻辑")
    print("=" * 60)
    
    # 示例数据
    ce_values = [1.0, 1.5, 2.0, 2.5]
    brier_values = [0.1, 0.3, 0.5, 0.7]
    evloss_values = [0.2, 0.8, 1.6, 3.2]
    
    print(f"\nCE values: {ce_values}")
    print(f"Brier values: {brier_values}")
    print(f"EVLoss values: {evloss_values}")
    
    # CE normalization
    ce_min = min(ce_values)
    ce_max = max(ce_values)
    ce_norm = [(c - ce_min) / (ce_max - ce_min) for c in ce_values]
    print(f"\nCE normalization:")
    print(f"  min = {ce_min}, max = {ce_max}")
    print(f"  normalized = {[f'{v:.3f}' for v in ce_norm]}")
    
    # Brier (already in [0, 1])
    brier_norm = brier_values
    print(f"\nBrier normalization:")
    print(f"  unchanged (already [0,1]) = {[f'{v:.3f}' for v in brier_norm]}")
    
    # EVLoss / 4
    evloss_norm = [e / 4.0 for e in evloss_values]
    print(f"\nEVLoss normalization:")
    print(f"  divided by 4 = {[f'{v:.3f}' for v in evloss_norm]}")
    
    # Union
    union_values = [(ce_norm[i] + brier_norm[i] + evloss_norm[i]) / 3.0 
                    for i in range(len(ce_values))]
    print(f"\nUnion Loss:")
    print(f"  (CE_norm + Brier_norm + EVLoss_norm) / 3")
    print(f"  = {[f'{v:.3f}' for v in union_values]}")
    print(f"  mean = {sum(union_values)/len(union_values):.3f}")


if __name__ == "__main__":
    test_cross_entropy()
    test_brier_score()
    test_ev_loss()
    test_union_logic()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
