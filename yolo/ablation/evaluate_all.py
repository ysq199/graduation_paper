"""
============================================================
统一评估脚本 — 对比所有 5 个消融实验步骤的最终指标
============================================================
评估每个步骤的 best.pt，记录：
- mask mAP50、mask mAP50-95
- Precision(M)、Recall(M)
- 参数量 (M)
- GFLOPs、FPS

输出：markdown 格式的对比表（方便直接复制到论文）
运行方式：python evaluate_all.py
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
BASE_RUNS_DIR = r"D:\projects\graduation_paper\yolo\yolov11\runs\segment"

STEPS = [
    {
        "name": "Step 1: Baseline",
        "tag": "YOLO11s-seg 原始",
        "weights": f"{BASE_RUNS_DIR}/ablation_step1_baseline/weights/best.pt",
    },
    {
        "name": "Step 2: +P2",
        "tag": "增加小目标分割分支",
        "weights": f"{BASE_RUNS_DIR}/ablation_step2_p2/weights/best.pt",
    },
    {
        "name": "Step 3: +P2+EMA",
        "tag": "加入 EMA 注意力",
        "weights": f"{BASE_RUNS_DIR}/ablation_step3_p2_ema/weights/best.pt",
    },
    {
        "name": "Step 4: +P2+EMA+BL",
        "tag": "加入 Boundary Loss",
        "weights": f"{BASE_RUNS_DIR}/ablation_step4_p2_ema_bl/weights/best.pt",
    },
    {
        "name": "Step 5: +数据增强",
        "tag": "强反光/模糊/低照度增强",
        "weights": f"{BASE_RUNS_DIR}/ablation_step5_augment/weights/best.pt",
    },
]


def get_gflops(model, input_size=800):
    """估算 GFLOPs（用 thop 或粗略估算）"""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, input_size, input_size).to(DEVICE)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        return None  # thop 未安装，用参数近似估算


def measure_fps(model, input_size=800, warmup=10, runs=100):
    """测试 FPS"""
    dummy = torch.randn(1, 3, input_size, input_size).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
    return runs / elapsed


def main():
    results = []

    for step in STEPS:
        print(f"\n{'='*60}")
        print(f"评估: {step['name']}")
        print(f"{'='*60}")

        weights_path = step['weights']

        if not Path(weights_path).exists():
            print(f"  [警告] 权重文件不存在: {weights_path}")
            print(f"  [跳过] 请先训练 Step {step['name']}")
            results.append({
                "name": step['name'],
                "tag": step['tag'],
                "map50": "-",
                "map50_95": "-",
                "precision": "-",
                "recall": "-",
                "params": "-",
                "gflops": "-",
                "fps": "-",
            })
            continue

        # 加载模型
        model = YOLO(weights_path)

        # 验证集评估
        metrics = model.val(data=DATA_YAML, device=DEVICE, imgsz=800, split='val',
                            verbose=False)

        map50 = metrics.seg.map50
        map50_95 = metrics.seg.map
        precision = metrics.seg.mp
        recall = metrics.seg.mr

        # 参数量
        params_M = sum(p.numel() for p in model.model.parameters()) / 1e6

        # GFLOPs
        gflops = get_gflops(model.model)
        gflops_str = f"{gflops:.1f}" if gflops else "N/A"

        # FPS
        fps = measure_fps(model.model)
        fps_str = f"{fps:.1f}"

        results.append({
            "name": step['name'],
            "tag": step['tag'],
            "map50": f"{map50:.4f}",
            "map50_95": f"{map50_95:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "params": f"{params_M:.2f}",
            "gflops": gflops_str,
            "fps": fps_str,
        })

        print(f"  mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, P={precision:.4f}, R={recall:.4f}")

    # ==================== 输出 Markdown 对比表 ====================
    print("\n\n")
    print("=" * 80)
    print("消融实验对比表（可直接复制到论文）")
    print("=" * 80)

    # 表头
    header = (
        "| 实验 | 说明 | mask mAP50 | mask mAP50-95 | Precision | Recall | 参数量(M) | GFLOPs | FPS |\n"
        "|------|------|-----------|--------------|-----------|--------|----------|--------|-----|"
    )
    print(header)

    for r in results:
        row = (
            f"| {r['name']} | {r['tag']} "
            f"| {r['map50']} | {r['map50_95']} "
            f"| {r['precision']} | {r['recall']} "
            f"| {r['params']} | {r['gflops']} | {r['fps']} |"
        )
        print(row)

    # ==================== 计算增量（Δ） ====================
    print("\n\n--- 消融增量分析 ---")
    if results and results[0]['map50'] != '-':
        baseline_map50 = float(results[0]['map50'])
        baseline_map95 = float(results[0]['map50_95'])
        for r in results[1:]:
            if r['map50'] != '-':
                delta50 = float(r['map50']) - baseline_map50
                delta95 = float(r['map50_95']) - baseline_map95
                print(f"{r['name']}  vs Baseline:")
                print(f"  Δ mask mAP50:    {delta50:+.4f}")
                print(f"  Δ mask mAP50-95: {delta95:+.4f}")

    # ==================== 保存到文件 ====================
    output_path = Path(__file__).parent / "ablation_results.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# YOLO11-seg 消融实验结果\n\n")
        f.write(f"数据集: Severstal Steel Defect\n")
        f.write(f"输入尺寸: 800\n")
        f.write(f"Epochs: 200\n\n")
        f.write(header + "\n")
        for r in results:
            row = (
                f"| {r['name']} | {r['tag']} "
                f"| {r['map50']} | {r['map50_95']} "
                f"| {r['precision']} | {r['recall']} "
                f"| {r['params']} | {r['gflops']} | {r['fps']} |"
            )
            f.write(row + "\n")

    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
