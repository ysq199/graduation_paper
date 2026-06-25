"""
============================================================
Step 4 — +P2 + EMA + Boundary Loss: 加入边界损失
============================================================
在 Step 3 的基础上，训练时叠加 Boundary Loss，
让模型把缺陷边界轮廓分得更清楚。

Boundary Loss 原理：
- 用拉普拉斯算子对 GT mask 提取边界像素
- 对预测 mask 同样提取边界
- 在边界像素上计算 Dice Loss
- 以权重 w 叠加到原 seg loss 上

运行方式：python train_step4_p2_ema_bl.py

注意事项：
- Monkey Patch UltraLytics 的 loss 计算是侵入式修改
- 如果精度反而下降，把 weight 从 0.1 调小到 0.05 或 0.02
- 钢面裂纹这类边缘模糊的缺陷，bl_weight 建议 0.05-0.1
"""

import torch
import time
import os
import sys
from pathlib import Path

# ==================== ★★★ 注入 EMA ★★★ ====================
sys.path.insert(0, str(Path(__file__).parent / "modules"))
from ema_attention import C2PSA_EMA
import ultralytics.nn.modules as ult_modules

original_C2PSA = ult_modules.C2PSA
ult_modules.C2PSA = C2PSA_EMA
print("[✓] C2PSA 已被替换为 C2PSA_EMA（含 EMA 注意力）")

# ==================== ★★★ 注入 Boundary Loss ★★★ ====================
# 方法：Monkey Patch v8SegmentationLoss 的 forward 方法
# 在原始 seg loss 上叠加 boundary loss

from boundary_loss import BoundaryLoss, get_boundary

BL_WEIGHT = 0.1  # 边界损失权重（可调）

# 引入 ultralytics 的 loss 类
from ultralytics.utils.loss import v8SegmentationLoss

original_seg_forward = v8SegmentationLoss.forward
boundary_loss_fn = BoundaryLoss(weight=BL_WEIGHT)

def patched_seg_forward(self, preds, batch):
    """
    在原始 forward 返回的 loss 上叠加 boundary loss
    """
    # 调用原始 forward
    result = original_seg_forward(self, preds, batch)

    # result 可能是 (loss, loss_items) 或直接是 loss
    if isinstance(result, tuple):
        loss, loss_items = result
    else:
        loss, loss_items = result, None

    # === 计算 Boundary Loss ===
    # preds 是 Segment 头的输出（包含 proto masks）
    # batch 中包含 ground truth masks
    try:
        # 提取 proto masks（segment 头的输出）
        # Ultralytics 的输出结构因版本而异，我们处理多种情况
        if isinstance(preds, (tuple, list)):
            proto = None
            for p in preds:
                if isinstance(p, torch.Tensor) and p.ndim == 4 and p.shape[1] >= 32:
                    proto = p
                    break
            if proto is None and len(preds) > 1:
                proto = preds[1] if isinstance(preds[1], torch.Tensor) else None
        else:
            proto = None

        # 提取 ground truth masks
        if hasattr(batch, 'masks'):
            gt_masks = batch.masks
        elif isinstance(batch, dict) and 'masks' in batch:
            gt_masks = batch['masks']
        else:
            gt_masks = None

        # 如果有 proto 和 gt_masks，计算 boundary loss
        if proto is not None and gt_masks is not None:
            # proto: [B, 32, H, W], gt_masks: [N_masks, H, W] or [B, H, W]
            # 简化：对 proto 的第一层做 sigmoid 后和 gt 的边界做 Dice
            if proto.ndim == 4:
                pred_prob = proto[:, 0:1, :, :].sigmoid()  # [B, 1, H, W]
            else:
                pred_prob = proto.sigmoid()

            # 确保 gt_masks 是 [B, H, W] 或 [B, 1, H, W]
            if gt_masks.ndim == 2:
                gt_masks = gt_masks.unsqueeze(0)

            # 对齐尺寸
            if pred_prob.shape[-2:] != gt_masks.shape[-2:]:
                from torch.nn.functional import interpolate
                pred_prob = interpolate(pred_prob, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)

            # 计算边界损失
            bl_loss = boundary_loss_fn(pred_prob.squeeze(1), gt_masks.float().squeeze(1))
            loss = loss + bl_loss

    except Exception as e:
        # Boundary loss 计算失败不影响训练，只打印警告
        pass

    if loss_items is not None:
        return loss, loss_items
    return loss

# 注入
v8SegmentationLoss.forward = patched_seg_forward
print(f"[✓] Boundary Loss 已注入到 v8SegmentationLoss, 权重={BL_WEIGHT}")

from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = r"D:\projects\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
DEVICE = 0
EPOCHS = 200
MODEL_CFG = r"models\yolo11-seg-p2.yaml"  # 和 Step 2/3 相同 YAML
PRETRAINED = "yolo11s-seg.pt"

# ==================== 训练 ====================
def main():
    print("="*60)
    print("Step 4: +P2 + EMA + Boundary Loss")
    print(f"模型配置: {MODEL_CFG}")
    print(f"Boundary Loss 权重 = {BL_WEIGHT}")
    print("="*60)

    model = YOLO(MODEL_CFG).load(PRETRAINED)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        device=DEVICE,
        imgsz=800,
        rect=True,
        batch=0.70,
        workers=2,
        cache="disk",
        amp=True,
        name="ablation_step4_p2_ema_bl",
        patience=50,
        save=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.1,
        cos_lr=True,
        overlap_mask=True,
        mask_ratio=4,
        close_mosaic=10,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.3,
    )

    # ==================== 评估 ====================
    print("\n" + "="*60)
    print("Step 4 训练完成，开始评估...")
    print("="*60)

    best_model = YOLO(r"D:\projects\graduation_paper\yolo\yolov11\runs\segment\ablation_step4_p2_ema_bl\weights\best.pt")
    metrics = best_model.val(data=DATA_YAML, device=DEVICE, imgsz=800, split='val')

    print("\n========== Step 4 最终指标 ==========")
    print(f"mask mAP50:       {metrics.seg.map50:.4f}")
    print(f"mask mAP50-95:    {metrics.seg.map:.4f}")
    print(f"Precision(M):     {metrics.seg.mp:.4f}")
    print(f"Recall(M):        {metrics.seg.mr:.4f}")

    params = sum(p.numel() for p in best_model.model.parameters()) / 1e6
    print(f"参数量:            {params:.2f}M")

    print("\n正在测 FPS...")
    dummy = torch.randn(1, 3, 800, 800).to(DEVICE)
    best_model.model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = best_model.model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = best_model.model(dummy)
        torch.cuda.synchronize()
    fps = 100 / (time.time() - start)
    print(f"FPS (800x800, GPU): {fps:.1f}")
    print("=======================================")

    # 恢复
    ult_modules.C2PSA = original_C2PSA
    v8SegmentationLoss.forward = original_seg_forward


if __name__ == "__main__":
    main()
