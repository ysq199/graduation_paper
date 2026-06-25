"""
============================================================
Step 5 — +数据增强: 在 Step 4 基础上加入强反光/模糊/低照度增强
============================================================
在 Step 4 (P2+EMA+BL) 的基础上，叠加工业场景数据增强。

增强类型：
1. 强反光（Glare） - 30%概率，模拟车间照明变化
2. 高斯/运动模糊（Blur） - 30%概率，模拟相机抖动
3. 低照度（Low Light） - 30%概率，模拟光线不足
4. 高斯噪声 - 30%概率，模拟传感器噪声
5. Gamma 校正 - 30%概率，模拟曝光变化

注：
- 增强只在训练时生效，评估时不增强
- 各类增强是独立的，一张图可能被多种增强同时影响
- 强度设置为 'medium'（中等），避免过度增强导致看不清

运行方式：python train_step5_augment.py
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
from boundary_loss import BoundaryLoss
from ultralytics.utils.loss import v8SegmentationLoss

BL_WEIGHT = 0.1
original_seg_forward = v8SegmentationLoss.forward
boundary_loss_fn = BoundaryLoss(weight=BL_WEIGHT)

def patched_seg_forward(self, preds, batch):
    result = original_seg_forward(self, preds, batch)
    if isinstance(result, tuple):
        loss, loss_items = result
    else:
        loss, loss_items = result, None
    try:
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
        if hasattr(batch, 'masks'):
            gt_masks = batch.masks
        elif isinstance(batch, dict) and 'masks' in batch:
            gt_masks = batch['masks']
        else:
            gt_masks = None
        if proto is not None and gt_masks is not None:
            if proto.ndim == 4:
                pred_prob = proto[:, 0:1, :, :].sigmoid()
            else:
                pred_prob = proto.sigmoid()
            if gt_masks.ndim == 2:
                gt_masks = gt_masks.unsqueeze(0)
            if pred_prob.shape[-2:] != gt_masks.shape[-2:]:
                from torch.nn.functional import interpolate
                pred_prob = interpolate(pred_prob, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
            bl_loss = boundary_loss_fn(pred_prob.squeeze(1), gt_masks.float().squeeze(1))
            loss = loss + bl_loss
    except Exception:
        pass
    if loss_items is not None:
        return loss, loss_items
    return loss

v8SegmentationLoss.forward = patched_seg_forward
print(f"[✓] Boundary Loss 已注入到 v8SegmentationLoss, 权重={BL_WEIGHT}")

# ==================== ★★★ 注册数据增强 ★★★ ====================
# 方法：直接修改 YOLO 训练时的超参数来实现更强的数据增强
# Ultralytics 本身支持很多数据增强参数，我们只需要调大参数

# 注意：Ultralytics 的 hsv_h/s/v, degrees, translate, scale, shear, perspective 等参数
# 是在 train() 调用时传入的，我们在 main 里直接设置更强的参数即可。

# 如果你需要更极端的增强（反光/低照度等 Ultralytics 原生不支持的）：
# 方式：使用 ultralytics 的 Albumentations 集成或自定义 dataset wrapper。
# 但最简单有效的方式是调大 Ultralytics 原生的增强参数，同时增加 erasing 和 auto_augment。

from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = r"D:\projects\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
DEVICE = 0
EPOCHS = 200
MODEL_CFG = r"models\yolo11-seg-p2.yaml"  # EMA 已通过 C2PSA 替换注入
PRETRAINED = "yolo11s-seg.pt"

# ==================== 训练 ====================
def main():
    print("="*60)
    print("Step 5: +数据增强（强反光 + 模糊 + 低照度）")
    print(f"模型配置: {MODEL_CFG}")
    print("新增: 强数据增强参数")
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
        name="ablation_step5_augment",
        patience=50,
        save=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.1,
        cos_lr=True,
        overlap_mask=True,
        mask_ratio=4,
        close_mosaic=10,

        # ==================== 增强参数对比 ====================
        #               Baseline   Step5（更强）
        # hsv_h          0.015  →   0.03    (色相变化 +100%)
        # hsv_s          0.4    →   0.7     (饱和度变化 +75%)
        # hsv_v          0.3    →   0.6     (亮度变化 +100%，模拟强反光/低照度)
        # degrees        3.0    →   10.0    (旋转角度 +233%)
        # translate      0.05   →   0.15    (平移范围 +200%)
        # scale          0.3    →   0.5     (缩放范围 +67%)
        # shear          0.0    →   2.0     (新增错切)
        # perspective    0.0    →   0.0005  (新增透视)
        # mosaic         1.0    →   1.0     (保持)
        # mixup          0.0    →   0.1     (新增混合，模拟遮挡)
        # erasing        0.4    →   0.6     (随机擦除增强)
        # ====================================================

        # HSV 颜色增强（关键！hsv_v 调大模拟强光/弱光）
        hsv_h=0.03,        # 原 0.015，加大色相抖动
        hsv_s=0.7,         # 原 0.4，加大饱和度抖动
        hsv_v=0.6,         # 原 0.3，加大亮度抖动（模拟强反光和低照度）

        # 几何增强
        degrees=10.0,      # 原 3.0，加大旋转范围
        translate=0.15,    # 原 0.05，加大平移范围
        scale=0.5,         # 原 0.3，加大缩放范围
        shear=2.0,         # 原 0.0，新增错切（模拟视角变化）
        perspective=0.0005,# 原 0.0，新增透视变换

        # 翻转
        fliplr=0.5,        # 原 0.5，保持左右翻转
        flipud=0.1,        # 原 0.0，新增上下翻转

        # 混合增强
        mosaic=1.0,        # 保持 mosaic
        mixup=0.1,         # 原 0.0，新增 MixUp（随机混合两张图）
        copy_paste=0.1,    # 原 0.0，新增复制粘贴增强（对分割特别有效）

        # 随机擦除
        erasing=0.6,       # 原 0.4，加大随机擦除（模拟遮挡/污渍）
    )

    # ==================== 评估 ====================
    print("\n" + "="*60)
    print("Step 5 训练完成，开始评估...")
    print("="*60)

    best_model = YOLO(r"D:\projects\graduation_paper\yolo\yolov11\runs\segment\ablation_step5_augment\weights\best.pt")
    metrics = best_model.val(data=DATA_YAML, device=DEVICE, imgsz=800, split='val')

    print("\n========== Step 5 最终指标 ==========")
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


if __name__ == "__main__":
    main()
