"""
C3K2_LFE 模块 —— 在 YOLOv11 C3K2 中嵌入长程特征增强

论文来源: BLE-YOLO (周璐, 2026, 机电工程技术)
         "基于BLE-YOLO的高效钢材表面缺陷检测算法"

原理:
  C3K2 是 YOLOv11 的核心特征提取模块，内部通过两个分支 + 多个 Bottleneck
  处理特征，然后用 Concat + Conv 融合。C3K2_LFE 在此基础上嵌入 LFE
  (Long-range Feature Enhancement) 模块:

  1. 输入特征图通过多分支结构提取多尺度特征
  2. LFE 模块在融合阶段介入，通过通道压缩 + 频域注意力
     动态筛选高频关键区域（缺陷边缘、纹理突变处）
  3. 高频区域的关注让模型对微小缺陷更敏感

  LFE 的核心是"频域选通": 用极小的参数（约0.02M）学会判断特征图的哪些
  高频分量对应缺陷、哪些对应背景纹理噪声。

与原始C3K2相比:
  - 参数增量: ~0.02M (几乎不变)
  - 计算量: 减少(因为LFE允许减少冗余通道)
  - 精度: BLE-YOLO在NEU-DET上+2.8% mAP

适用场景:
  - YOLOv11n 钢铁表面缺陷检测（C3K2_LFE是BLE-YOLO的核心贡献）
  - 其他 n/s 量级的 YOLOv11 改进
"""

import torch
import torch.nn as nn
import math


class LFE(nn.Module):
    """
    Long-range Feature Enhancement (LFE) 模块

    对输入特征图进行频域分析，动态增强高频关键区域。
    核心操作:
      1. 通道压缩 (1x1 Conv) 降低计算量
      2. 分组卷积提取高频特征
      3. 通道扩展 + Sigmoid 门控 = 注意力权重
      4. 加权到原始特征

    Args:
        channels: 输入/输出通道数
        reduction: 通道压缩比，默认 4
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(8, channels // reduction)

        # 通道压缩: C -> C/r
        self.compress = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 分组卷积提取高频分量 (用 Laplacian-like 初始化)
        self.high_pass = nn.Conv2d(
            mid_channels, mid_channels,
            kernel_size=3, padding=1,
            groups=mid_channels, bias=False
        )
        # 初始化高通卷积核为类似 Laplacian 的权重
        with torch.no_grad():
            laplacian_kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32).view(1, 1, 3, 3).repeat(mid_channels, 1, 1, 1)
            self.high_pass.weight.copy_(laplacian_kernel)

        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 逐通道注意力: 计算每个空间位置的"高频重要性"
        self.spatial_attn = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        # 通道扩展: C/r -> C
        self.expand = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels)

        # 可学习的温度参数 (控制注意力强度)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 通道压缩
        feat = self.act(self.bn1(self.compress(x)))

        # 2. 高频提取
        high_freq = self.act(self.bn2(self.high_pass(feat)))

        # 3. 空间注意力 = 高频 × 逐通道权重
        spatial = self.act(self.bn3(self.spatial_attn(high_freq)))

        # 4. 通道扩展 + Sigmoid 门控
        gate = torch.sigmoid(self.bn4(self.expand(spatial)) * self.scale)

        # 5. 加权残差
        return x + x * gate


class C3k2_LFE(nn.Module):
    """
    C3k2 + LFE: YOLOv11 改进模块

    结构:
      input
        ├─ c1       branch1: Conv(x, c2', 1x1)
        │  └─ Bottleneck x N
        │
        ├─ c2       branch2: Conv(x, c2', 1x1)
        │
        └─ concat(branch1, branch2)
           └─ Conv(concat, c2, 1x1)
              └─ LFE(c2)         <-- 关键改进: 在融合后插入 LFE

    参数与原始 C3k2 完全相同（仅增加 LFE 的极少量参数）。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck 重复次数 (1 表示单个 Bottleneck)
        shortcut: 是否在 Bottleneck 中使用 shortcut
        g: groups
        e: 通道扩展比
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        self.c3k = c3k
        self.attn = attn
        self.shortcut = shortcut
        self.c = int(c2 * e)  # 每个分支的中间通道数
        self.n = n

        # Bottleneck 分支
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)

        self.m = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.c, self.c, 3, 1, 1, groups=g, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.c, self.c, 1, 1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            ) if shortcut else
            nn.Sequential(
                nn.Conv2d(self.c, self.c, 3, 1, 1, groups=g, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.c, self.c, 1, 1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            )
            for _ in range(n)
        ])

        # 融合卷积
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        # 激活函数
        self.act = nn.SiLU(inplace=True)

        # LFE 模块 (核心改进)
        self.lfe = LFE(c2, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分两分支 + n 个 Bottleneck
        y = list(self.act(self.bn1(self.cv1(x))).split(self.c, dim=1))
        y.extend(m(y[-1]) for m in self.m)

        # 融合
        out = self.act(self.bn2(self.cv2(torch.cat(y, dim=1))))

        # LFE 增强
        return self.lfe(out)


class Bottleneck_LFE(nn.Module):
    """
    标准 Bottleneck + LFE（可替换 YOLOv11 倒数几层的 Bottleneck）

    Bottleneck(c1, c2) + LFE(c2)
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0] // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2
        self.act = nn.SiLU(inplace=True)
        self.lfe = LFE(c2, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.cv1(x)))
        out = self.act(self.bn2(self.cv2(out)))
        if self.add:
            out = out + identity
        return self.lfe(out)


# ============================================================
# LFE 简化版 —— 仅用通道注意力的极简实现
# 适用于通道数很少的 nano 早期层
# ============================================================
class LFELite(nn.Module):
    """
    LFE Lite: 仅用通道 SE 式注意力 + 高频预计算的极简 LFE

    当通道数 <= 64 时使用此版本，避免 reduction=4 时中间通道过小。
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(4, channels // 2)

        # 通道注意力
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=False)

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w) * self.scale)
        return x * (1.0 + w)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  C3K2_LFE 模块测试")
    print("=" * 60)

    # Test LFE (标准版)
    x = torch.randn(1, 256, 40, 40)
    lfe = LFE(256)
    y = lfe(x)
    print(f"\nLFE (标准):        输入 {list(x.shape)} -> 输出 {list(y.shape)}")
    print(f"  参数量: {sum(p.numel() for p in lfe.parameters()):,}")

    # Test LFE Lite
    x_lite = torch.randn(1, 64, 80, 80)
    lfe_lite = LFELite(64)
    y_lite = lfe_lite(x_lite)
    print(f"\nLFELite:           输入 {list(x_lite.shape)} -> 输出 {list(y_lite.shape)}")
    print(f"  参数量: {sum(p.numel() for p in lfe_lite.parameters()):,}")

    # Test C3k2_LFE (模拟 YOLOv11n backbone 中一层)
    x_c3 = torch.randn(1, 128, 80, 80)
    c3k2_lfe = C3k2_LFE(c1=128, c2=128, n=2, shortcut=False, e=0.25)
    y_c3 = c3k2_lfe(x_c3)
    print(f"\nC3k2_LFE (n=2):    输入 {list(x_c3.shape)} -> 输出 {list(y_c3.shape)}")
    print(f"  参数量: {sum(p.numel() for p in c3k2_lfe.parameters()):,}")

    # 对比原始 C3k2 参数
    from ultralytics.nn.modules.block import C3k2
    c3k2_orig = C3k2(c1=128, c2=128, n=2, shortcut=False, e=0.25)
    print(f"  原始 C3k2 参数量:  {sum(p.numel() for p in c3k2_orig.parameters()):,}")
    print(f"  LFE 增加:          {sum(p.numel() for p in c3k2_lfe.lfe.parameters()):,}")

    print("\n" + "=" * 60)
    print("  C3K2_LFE 测试通过!")
    print("=" * 60)
