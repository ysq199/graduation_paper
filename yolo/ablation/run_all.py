"""
============================================================
一键运行所有消融实验（依次训练 5 个步骤）
============================================================
适合一次性跑完所有实验，自动记录结果。

使用方式：
    python run_all.py

注意：
- 每步训练 200 epochs，全部跑完约需 15~40 小时（取决于 GPU）
- 建议先单独跑 Step 1 验证环境，再跑全量
- 如果 GPU 显存不足，把 batch 从 0.70 减小到 0.50

数据集配置：
- 默认使用 Severstal Steel Defect 数据集
- 如需修改，改下面的 DATA_YAML 变量
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
STEPS = [
    ("train_step1_baseline.py", "Step 1: Baseline"),
    ("train_step2_p2.py", "Step 2: +P2"),
    ("train_step3_p2_ema.py", "Step 3: +P2+EMA"),
    ("train_step4_p2_ema_bl.py", "Step 4: +P2+EMA+BL"),
    ("train_step5_augment.py", "Step 5: +数据增强"),
]


def run_all():
    for script, description in STEPS:
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            print(f"[错误] 脚本不存在: {script_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"# {description}")
        print(f"{'#'*60}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPTS_DIR),
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            print(f"[失败] {description} 执行失败，返回码: {result.returncode}")
            # 可以选择跳过后续步骤或终止
            response = input("\n继续执行下一步? (y/n): ")
            if response.lower() != 'y':
                break

    print("\n" + "="*60)
    print("所有步骤完成！现在运行评估...")
    print("="*60)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / "evaluate_all.py")],
                   cwd=str(SCRIPTS_DIR))


if __name__ == "__main__":
    run_all()
