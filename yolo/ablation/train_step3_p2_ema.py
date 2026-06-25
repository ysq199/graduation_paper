"""
============================================================
Step 3 — +P2 + EMA: 最简单的 EMA 注入方式
============================================================
策略：直接替换 Ultralytics 中的 C2PSA 类 → C2PSA_EMA
然后使用和 Step 2 完全相同的 YAML（yolo11-seg-p2.yaml）

不需要创建新的 yaml 文件，不需要复杂注册，
只需要在 import YOLO 之前替换类即可。

运行方式：python train_step3_p2_ema.py
"""

import torch
import time
import os
import sys
from pathlib import Path

# ==================== ★★★ 注入 EMA ★★★ ====================
# 必须在 import ultralytics 之前完成
sys.path.insert(0, str(Path(__file__).parent / "modules"))
from ema_attention import C2PSA_EMA
import ultralytics.nn.modules as ult_modules

# 保存原始 C2PSA，以便后续恢复（如果不需要可以忽略）
original_C2PSA = ult_modules.C2PSA
ult_modules.C2PSA = C2PSA_EMA  # 全局替换！
print("[✓] C2PSA 已被替换为 C2PSA_EMA（含 EMA 注意力）")

from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = r"D:\projects\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
DEVICE = 0
EPOCHS = 200
# 和 Step 2 使用相同的 YAML！（因为 C2PSA 类已被替换）
MODEL_CFG = r"models\yolo11-seg-p2.yaml"
PRETRAINED = "yolo11s-seg.pt"

# ==================== 训练 ====================
def main():
    print("="*60)
    print("Step 3: +P2 + EMA 轻量注意力")
    print(f"模型配置: {MODEL_CFG}")
    print("策略: 全局替换 C2PSA → C2PSA_EMA")
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
        name="ablation_step3_p2_ema",
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
    print("Step 3 +P2+EMA 训练完成，开始评估...")
    print("="*60)

    best_model = YOLO(r"D:\projects\graduation_paper\yolo\yolov11\runs\segment\ablation_step3_p2_ema\weights\best.pt")
    metrics = best_model.val(data=DATA_YAML, device=DEVICE, imgsz=800, split='val')

    print("\n========== Step 3 +P2+EMA 最终指标 ==========")
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
    print("==============================================")

    # 恢复原始 C2PSA（可选）
    ult_modules.C2PSA = original_C2PSA


if __name__ == "__main__":
    main()
