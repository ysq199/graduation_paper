"""
============================================================
Step 1 — Baseline: 原始 YOLO11n/s-seg
============================================================
不做任何改动，直接用官方预训练权重训练实例分割。
记录 mask mAP50、mask mAP50-95、Precision、Recall、参数量、FPS。

运行方式：python train_step1_baseline.py
"""

import torch
import time
from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = r"D:\projects\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
DEVICE = 0
EPOCHS = 200

# ==================== 训练 ====================
def main():
    # 创建模型（加载官方预训练权重）
    model = YOLO("yolo11s-seg.pt")  # 或 "yolo11n-seg.pt"

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
        name="ablation_step1_baseline",
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

    # ==================== 评估（记录指标） ====================
    print("\n" + "="*60)
    print("Step 1 Baseline 训练完成，开始评估...")
    print("="*60)

    # 加载 best.pt
    best_model = YOLO(r"D:\projects\graduation_paper\yolo\yolov11\runs\segment\ablation_step1_baseline\weights\best.pt")

    # 在验证集上评估
    metrics = best_model.val(data=DATA_YAML, device=DEVICE, imgsz=800, split='val')

    print("\n========== Step 1 Baseline 最终指标 ==========")
    print(f"mask mAP50:       {metrics.seg.map50:.4f}")
    print(f"mask mAP50-95:    {metrics.seg.map:.4f}")
    print(f"Precision(M):     {metrics.seg.mp:.4f}")
    print(f"Recall(M):        {metrics.seg.mr:.4f}")

    # 参数量
    params = sum(p.numel() for p in best_model.model.parameters()) / 1e6
    print(f"参数量:            {params:.2f}M")

    # FPS（GPU 预热后测 100 次取平均）
    print("\n正在测 FPS...")
    dummy = torch.randn(1, 3, 800, 800).to(DEVICE)
    best_model.model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = best_model.model(dummy)
        # 计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = best_model.model(dummy)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    fps = 100 / elapsed
    print(f"FPS (800x800, GPU): {fps:.1f}")
    print("==============================================")


if __name__ == "__main__":
    main()
