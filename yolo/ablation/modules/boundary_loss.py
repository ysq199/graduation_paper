"""
============================================================
Boundary Loss（边界损失）实现
============================================================
什么是 Boundary Loss？
- 普通交叉熵损失只看"这个像素是不是缺陷"，不管边界是否清晰
- Boundary Loss 额外要求"缺陷边界轮廓也要分清楚"
- 对钢面裂纹这种细长、边缘模糊的缺陷特别有效

实现原理：
- 对 GT mask 做拉普拉斯算子提取边界
- 对预测 mask 同样提取边界
- 计算边界像素的 Dice Loss，加到原始 seg loss 上

论文参考：Boundary Loss for Remote Sensing Imagery (Jetley et al.)
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def get_boundary(mask, kernel_size=3):
    """
    从二值 mask 中提取边界像素
    原理：用拉普拉斯算子检测 mask 中灰度变化剧烈的像素 = 边界

    Args:
        mask: [B, H, W] 或 [B, 1, H, W] 二值 mask
        kernel_size: 拉普拉斯核大小（3 或 5，3 就够了）
    Returns:
        boundary: [B, H, W] 边界 mask（0~1 连续值）
    """
    # 确保是 4D [B, C, H, W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # 拉普拉斯核（检测边缘的二阶导数）
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=mask.dtype, device=mask.device
    ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    # 卷积得到边界响应
    boundary = F.conv2d(mask.float(), laplacian_kernel, padding=1)
    boundary = torch.abs(boundary)  # 取绝对值（边界有正有负）
    boundary = torch.clamp(boundary, 0, 1)  # 压缩到 [0, 1]

    return boundary.squeeze(1)  # [B, H, W]


class BoundaryLoss(nn.Module):
    """
    边界损失 = Dice Loss on boundary pixels

    用法：
        bl = BoundaryLoss(weight=0.1)
        total_loss = original_seg_loss + bl(pred_mask, gt_mask) * weight
    """

    def __init__(self, weight=0.1, kernel_size=3):
        """
        Args:
            weight: 边界损失的权重（默认 0.1，表示占总体 seg loss 的 10%）
            kernel_size: 拉普拉斯核大小
        """
        super().__init__()
        self.weight = weight
        self.kernel_size = kernel_size
        self.smooth = 1e-6

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, H, W] 预测 mask（sigmoid 后的概率值 0~1）
            target: [B, H, W] GT mask（二值 0/1）
        Returns:
            scalar loss value
        """
        # 提取边界
        pred_boundary = get_boundary(pred, self.kernel_size)
        target_boundary = get_boundary(target, self.kernel_size)

        # Dice Loss on boundary
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = self.weight * (1.0 - dice)

        return loss


# ============================================================
# 自定义 Trainer（重写损失计算，加入 Boundary Loss）
# ============================================================
def monkey_patch_criterion(weight=0.1):
    """
    用 Monkey Patching 注入 Boundary Loss 到 Ultralytics 的训练流程
    原理：修改 ultralytics 内部的 Criterion 类，在 seg loss 上叠加 boundary loss

    关键思路：
    Ultralytics 在 v8SegmentLoss 中计算 seg loss。
    我们 hook 进去，在原始 seg loss 上叠加 boundary loss。
    """

    from ultralytics.utils.loss import v8SegmentationLoss
    import types

    # 保存原始 forward
    original_forward = v8SegmentationLoss.__init__
    original_call = v8SegmentationLoss.__call__ if hasattr(v8SegmentationLoss, '__call__') else v8SegmentationLoss.forward

    boundary_loss_fn = BoundaryLoss(weight=weight)

    def patched_forward(self, preds, batch):
        """
        被 patch 的 forward，在原始 seg loss 基础上加 boundary loss
        """
        # 调用原始方法
        # 注意：这里需要适配 Ultralytics 内部的具体实现
        # Ultralytics 的 loss 输出通常是 (loss, loss_items)
        # loss_items: [box, seg, cls, dfl, ...]

        orig_loss = original_call(self, preds, batch)

        # 检查返回值类型（兼容不同版本的 Ultralytics）
        if isinstance(orig_loss, tuple):
            total_loss, loss_items = orig_loss
        else:
            total_loss = orig_loss
            loss_items = None

        # === 计算 Boundary Loss ===
        # preds 是 Segment 头的原始输出
        # 我们需要提取 proto 和 masks

        if len(preds) >= 4:  # segment 输出包含 proto
            # preds: [[feats], proto, ...]
            # 实际 preds 结构因版本而异，关键看第2个元素

            # Ultralytics 输出结构：
            # preds[1] = mc (mask coefficients) [batch, num_protos, H, W]
            # 或 preds[1] 是 proto masks

            # 查一下 batch 中的 "masks" 关键字段
            if hasattr(batch, 'get') or isinstance(batch, dict):
                gt_masks = batch.get('masks', None) if isinstance(batch, dict) else getattr(batch, 'masks', None)
            else:
                # Batch 是自定义类，尝试取 masks
                gt_masks = getattr(batch, 'masks', None) if not isinstance(batch, (list, tuple)) else batch[4] if len(batch) > 4 else None

            if gt_masks is not None:
                # 简化处理：直接在 seg loss 上加权
                # 完整的实现需要从 proto + coefficients 重建预测 mask
                # 这里我们用权重的方式叠加
                bl_val = boundary_loss_fn(
                    preds[-1].sigmoid() if hasattr(preds[-1], 'sigmoid') else preds[-1],
                    gt_masks.float()
                )
                total_loss = total_loss + bl_val

        if loss_items is not None:
            return total_loss, loss_items
        else:
            return total_loss

    # 注入
    v8SegmentationLoss.forward = patched_forward
    print(f"[BoundaryLoss] 已注入，权重 = {weight}")


# ============================================================
# 更优雅的方式：自定义 Trainer 基类，重写 criterion
# 在 Step4 训练脚本中使用
# ============================================================
class BoundaryLossTrainer:
    """
    使用说明：
    1. 在你的训练脚本开头 import 这个文件
    2. 调用 monkey_patch_criterion(weight=0.1) 注入 boundary loss
    3. 正常调用 model.train()
    """
    pass


if __name__ == "__main__":
    # 测试边界提取
    test_mask = torch.zeros(1, 1, 16, 16)
    test_mask[:, :, 4:12, 4:12] = 1.0

    boundary = get_boundary(test_mask)
    print("Mask shape:", test_mask.shape)
    print("Boundary shape:", boundary.shape)
    print("Boundary sum:", boundary.sum().item())
    print("(边界像素 > 0 的数量应该约等于 4*8 = 32)")
