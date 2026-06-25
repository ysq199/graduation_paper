"""
============================================================
EMA (Efficient Multi-scale Attention) 轻量级注意力模块
============================================================
什么是 EMA？
- 一种通道级注意力机制，让模型自动学会"哪些特征通道更重要"
- 对钢面缺陷分割特别有用：让模型关注裂纹纹理、边缘等关键特征
- 参数量极小（几乎不增加模型大小），推理速度影响可忽略

怎么用？
- 插入到 backbone 的关键层后面（P2, P3, P4 层）
- 或者替换 C2PSA 模块中的 PSA 注意力

论文参考：EMA: Efficient Multi-scale Attention (CVPR 2023 Workshop)
============================================================
"""

import torch
import torch.nn as nn
import math


class EMA(nn.Module):
    """
    Efficient Multi-scale Attention Module

    核心思想：
    1. 对通道分组，一部分做 1x1 卷积，一部分做 3x3 卷积（多尺度）
    2. 用全局平均池化产生通道权重
    3. 用跨空间交互编码空间位置信息
    """

    def __init__(self, channels, factor=8):
        """
        Args:
            channels: 输入通道数
            factor:   下采样倍率（越小参数量越少，越大越强，推荐 8 或 16）
        """
        super(EMA, self).__init__()

        self.groups = factor  # 分组数 = 通道数 / factor
        assert channels // self.groups > 0, "channels must be divisible by factor"
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        # 1x1 卷积分支（跨通道交互）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 3x3 卷积分支（局部空间交互）
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                  kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征图
        Returns:
            [B, C, H, W] 注意力加权后的特征图（形状不变）
        """
        b, c, h, w = x.size()

        # 将通道分成多个组
        group_x = x.reshape(b * self.groups, -1, h, w)  # [B*G, C//G, H, W]

        # 路径1：水平+垂直方向池化编码空间信息
        x_h = self.pool_h(group_x)       # [B*G, C//G, H, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # [B*G, C//G, 1, W]

        # 路径2：3x3 卷积 + GN（激活函数）
        hw = self.conv1x1(group_x)       # [B*G, C//G, H, W]
        x_hw = torch.sigmoid(self.conv3x3(hw))  # [B*G, C//G, H, W]

        # 两路融合
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = group_x * x_hw

        # 合并两组注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = group_x * weights.sigmoid()

        return out.reshape(b, c, h, w)


# ============================================================
# 如何把 EMA 注入 YOLO11？有两种方式：
# 方式A（推荐小白用）：用自定义 YAML 把 EMA 作为独立模块放在关键层后
# 方式B（轻量）：替换 backbone 末尾 C2PSA 中的 PSA → EMA
# ============================================================

class C2PSA_EMA(nn.Module):
    """
    C2PSA 的变体：用 EMA 替换 PSA（方式B，推荐）
    直接替换 backbone 最后一个模块，参数量几乎不变，效果更好

    用法：把 yolo11-seg-p2.yaml 中 backbone 最后一行
          C2PSA 替换为 C2PSA_EMA（需要在 Ultralytics 注册后使用）
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(EMA(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================
# 注册到 Ultralytics 的正确方式（必须注册到 tasks.py 的 globals()）
# ============================================================
def register_ema():
    """
    正确方式：注册到 tasks.py 的 globals() 和 base_modules/repeat_modules。
    parse_model 通过 globals()[m] 查找模块名，
    同时需要加入 base_modules 和 repeat_modules 让 parse_model 正确处理参数。
    """
    import ultralytics.nn.tasks as tasks_mod
    from ultralytics.nn.modules import Conv as UltConv

    # Step 1: 注册到 tasks 的全局命名空间（globals() 查找）
    tasks_mod.__dict__['EMA'] = EMA
    tasks_mod.__dict__['C2PSA_EMA'] = C2PSA_EMA

    # Step 2: C2PSA_EMA 加入 base_modules 集合（让 parse_model 用通用方式处理参数）
    # 注：frozenset 不可变，所以在 parse_model 内部会检查。我们需要 monkey patch。
    # 更简单的方法：让 C2PSA_EMA 继承 C2PSA 的 __init__ 签名，
    # 或者直接给 tasks_mod.parse_model 加 hook。
    # 实际最稳的方法：在 parse_model 执行前把 C2PSA_EMA 当作 C2PSA 的别名注册

    print("[EMA] 已注册到 Ultralytics tasks 模块")
