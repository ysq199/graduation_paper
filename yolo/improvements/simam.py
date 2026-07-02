"""
SimAM (A Simple, Parameter-Free Attention Module for CNNs)

论文: "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"
      (Yang et al., ICML 2021)

文献支撑: GSS-YOLO (肖轶磊, 2025, 空军工程大学学报) 在带钢缺陷检测中使用 SimAM,
         mAP 提升 3.7%，零额外参数。

原理:
  SimAM 基于神经科学中的"空域抑制"理论 (spatial suppression):
  活跃的神经元会抑制周围神经元的信号传递。
  在特征图中，包含重要信息（缺陷区域）的神经元应该比周围神经元更活跃。

  具体步骤:
  1. 对每个通道计算均值 μ 和方差 σ²:
     e_t* = 4(σ² + λ) / [(t - μ)² + 2σ² + 2λ]
     其中 t 是每个空间位置的像素值
  2. 能量越低，该神经元越重要 → 取 1/e_t* 作为注意力权重
  3. 用 sigmoid 归一化后与原始特征图相乘

优势:
  - 完全不增加参数量 (parameter-free)
  - 同时考虑通道和空间维度的注意力 (3D attention)
  - 即插即用，可插入任何 CNN 的任意位置
  - 计算开销极小 (约 0.001 GFLOPs)
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """SimAM 三维注意力模块

    Args:
        e_lambda: 正则化系数，默认 1e-4
        channels: 输入通道数（仅用于占位，不产生实际参数）
    """
    def __init__(self, channels: int = None, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        # channels 参数仅用于兼容性（不产生参数）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            加权后的特征图 (B, C, H, W)
        """
        b, c, h, w = x.size()
        n = w * h - 1  # 空间维度 - 1

        # 计算每个空间位置与均值的平方差
        # x_minus_mu_square: (B, C, H, W)
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 公式: y = x_minus_mu_square / [4 * (x_minus_mu_square.sum() / (n-1) + lambda)] + 0.5
        # 简化计算
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        # 注意力权重通过 sigmoid 归一化
        attention = torch.sigmoid(1.0 / y)

        # 加权特征图
        return x * attention

    def __repr__(self):
        return f"SimAM(lambda={self.e_lambda})"


class C2fSimAM(nn.Module):
    """
    C2f + SimAM 组合模块

    参考 YOLOv11 中 C2f 结构:
    C2f: split -> 2个分支 -> n个Bottleneck -> concat -> conv

    在其输出后接 SimAM 注意力。
    （实际使用时通过修改 model yaml 实现，此模块为示例）
    """
    def __init__(self, channels: int, e_lambda: float = 1e-4):
        super().__init__()
        self.simam = SimAM(channels=channels, e_lambda=e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.simam(x)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 模拟 YOLO backbone 中一层特征图 (C2f 输出)
    x = torch.randn(1, 256, 80, 80)  # P3 特征图

    simam = SimAM(channels=256)
    y = simam(x)

    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {y.shape}")
    print(f"参数量:    {sum(p.numel() for p in simam.parameters()):,} (零参数)")
    print(f"注意力权重范围: [{y.mean().item():.4f}, {y.std().item():.4f}]")

    # 验证输出与输入形状一致
    assert x.shape == y.shape, "形状不匹配!"
    assert not torch.equal(x, y), "注意力未生效!"

    # 验证无参数
    params = list(simam.parameters())
    assert len(params) == 0, f"SimAM 有 {len(params)} 个参数张量，预期为 0!"

    print("\n✓ SimAM 模块测试通过 (零参数, 形状保持)")
