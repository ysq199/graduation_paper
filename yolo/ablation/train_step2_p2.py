"""
============================================================
Step 2 — +P2: 增加 P2 小目标分割分支
============================================================
基于 yolo11-seg-p2.yaml 自定义模型配置训练。
P2 分支让模型能检测 4x4 像素以上的微小缺陷。

原理：
- 原始 YOLO 有 3 个检测头（P3/P4/P5），最小检测目标约 8x8 像素
- 加上 P2 后变成 4 个头，最小检测目标降到 4x4 像素
- 对钢面的微小裂纹、针孔缺陷特别有用

运行方式：python train_step2_p2.py
"""

import torch
import time
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = r"D:\projects\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
DEVICE = 0
EPOCHS = 200
MODEL_CFG = r"models\yolo11-seg-p2.yaml"   # P2 分支模型配置
PRETRAINED = "yolo11s-seg.pt"               # 用官方 seg 预训练权重初始化

# ==================== 训练 ====================
def main():
    print("="*60)
    print("Step 2: +P2 小目标分割分支")
    print(f"模型配置: {MODEL_CFG}")
    print(f"预训练权重: {PRETRAINED}")
    print("="*60)

    # 从 YAML 创建模型结构，加载预训练权重
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
        name="ablation_step2_p2",
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
    print("Step 2 +P2 训练完成，开始评估...")
    print("="*60)

    best_model = YOLO(r"D:\projects\graduation_paper\yolo\yolov11\runs\segment\ablation_step2_p2\weights\best.pt")
    metrics = best_model.val(data=DATA_YAML, device=DEVICE, imgsz=800, split='val')

    print("\n========== Step 2 +P2 最终指标 ==========")
    print(f"mask mAP50:       {metrics.seg.map50:.4f}")
    print(f"mask mAP50-95:    {metrics.seg.map:.4f}")
    print(f"Precision(M):     {metrics.seg.mp:.4f}")
    print(f"Recall(M):        {metrics.seg.mr:.4f}")

    params = sum(p.numel() for p in best_model.model.parameters()) / 1e6
    print(f"参数量:            {params:.2f}M")

    # FPS
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
    print("===========================================")


if __name__ == "__main__":
    main()
