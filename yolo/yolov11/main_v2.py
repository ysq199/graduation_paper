"""YOLOv11n v2 ablation runner for Severstal proxy experiments.

The real rotor dataset is not guaranteed yet, so this script treats the
Severstal steel-defect dataset as a proxy for small reflective steel defects.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

YOLOV11_DIR = Path(__file__).resolve().parent
YOLO_ROOT = YOLOV11_DIR.parent
IMPROVEMENTS_DIR = YOLO_ROOT / "improvements"
DEFAULT_DATA = YOLO_ROOT / "datasets" / "severstal-steel-defect-instance-segmentation.v4i.yolov11" / "data.yaml"
DEFAULT_PRETRAINED = YOLOV11_DIR / "yolo11n.pt"

sys.path.insert(0, str(YOLO_ROOT))

from improvements import patch_wise_iou_loss, register_improvements  # noqa: E402
from ultralytics import YOLO  # noqa: E402


# key -> (yaml file, use Wise-IoU patch, description)
MODEL_CONFIGS: dict[str, tuple[str, bool, str]] = {
    "baseline": ("yolo11n.yaml", False, "YOLOv11n baseline"),
    "simam": ("yolo11n-simam.yaml", False, "+ SimAM"),
    "lfe": ("yolo11n-lfe.yaml", False, "+ C3K2_LFE"),
    "lfe-simam": ("yolo11n-lfe-simam.yaml", False, "+ C3K2_LFE + SimAM"),
    "lfe-simam-slim": (
        "yolo11n-lfe-simam-slimneck.yaml",
        False,
        "+ C3K2_LFE + SimAM + Slim-Neck",
    ),
    "lfe-simam-slim-dysample": (
        "yolo11n-lfe-simam-slimneck-dysample.yaml",
        False,
        "+ C3K2_LFE + SimAM + Slim-Neck + DySample",
    ),
    "final": (
        "yolo11n-lfe-simam-slimneck-dysample.yaml",
        True,
        "+ C3K2_LFE + SimAM + Slim-Neck + DySample + Wise-IoU",
    ),
}

DEFAULT_EXPERIMENTS = [
    "baseline",
    "simam",
    "lfe",
    "lfe-simam",
    "lfe-simam-slim",
    "lfe-simam-slim-dysample",
    "final",
]

TRAIN_ARGS = {
    "epochs": 200,
    "device": 0,
    "imgsz": 640,
    "rect": True,
    "batch": 16,
    "workers": 4,
    "cache": "disk",
    "amp": True,
    "patience": 50,
    "save": True,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": True,
    "close_mosaic": 10,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "degrees": 3.0,
    "translate": 0.05,
    "scale": 0.3,
}


def resolve_model_path(yaml_file: str) -> str:
    """Return a local improvement yaml path, or an Ultralytics model name."""
    local_path = IMPROVEMENTS_DIR / yaml_file
    return str(local_path) if local_path.exists() else yaml_file


def build_model(mode: str, pretrained: str | Path | None = None) -> YOLO:
    """Build a YOLO model after registering custom v2 modules."""
    if mode not in MODEL_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Choices: {', '.join(MODEL_CONFIGS)}")

    yaml_file, use_wise_iou, _ = MODEL_CONFIGS[mode]
    register_improvements()
    if use_wise_iou:
        patch_wise_iou_loss()

    model = YOLO(resolve_model_path(yaml_file))
    if pretrained:
        model = model.load(str(pretrained))
    return model


def train_model(mode: str, data: str, pretrained: str, project: str | None = None) -> None:
    """Train one ablation configuration."""
    yaml_file, use_wise_iou, description = MODEL_CONFIGS[mode]
    model_path = resolve_model_path(yaml_file)
    print("\n" + "=" * 70)
    print(f"v2 experiment: {mode}")
    print(f"description : {description}")
    print(f"model yaml  : {model_path}")
    print(f"data yaml   : {data}")
    print(f"box loss    : {'Wise-IoU v2 patch' if use_wise_iou else 'Ultralytics default CIoU'}")
    print("=" * 70 + "\n")

    model = build_model(mode, pretrained=pretrained)
    train_args = TRAIN_ARGS.copy()
    if project:
        train_args["project"] = project

    model.train(data=data, name=f"v2_{mode}", **train_args)
    print(f"\n[done] {mode}: {description}")


def verify_modules() -> None:
    """Quickly verify module shapes and YAML model construction."""
    import torch

    register_improvements()

    from improvements import C3k2_LFE, DySample, LFE, LFELite, SimAM, VoVGSCSP

    checks = [
        ("C3K2_LFE", C3k2_LFE(128, 128, n=2, shortcut=False, e=0.25), torch.randn(1, 128, 80, 80)),
        ("LFE", LFE(256), torch.randn(1, 256, 40, 40)),
        ("LFELite", LFELite(64), torch.randn(1, 64, 80, 80)),
        ("SimAM", SimAM(), torch.randn(1, 256, 80, 80)),
        ("VoVGSCSP", VoVGSCSP(128, 128, n=1, shortcut=True, e=0.5), torch.randn(1, 128, 80, 80)),
        ("DySample", DySample(128, scale=2, groups=4), torch.randn(1, 128, 40, 40)),
    ]

    print("\n" + "=" * 70)
    print("v2 module shape checks")
    print("=" * 70)
    for name, module, x in checks:
        y = module(x)
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:12s}: {list(x.shape)} -> {list(y.shape)}, params={params:,}")

    print("\n" + "=" * 70)
    print("YAML construction checks")
    print("=" * 70)
    for mode in MODEL_CONFIGS:
        model = build_model(mode)
        params = sum(p.numel() for p in model.model.parameters())
        print(f"{mode:26s}: params={params:,}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv11n v2 ablation runner")
    parser.add_argument("--mode", default="baseline", choices=["all", *MODEL_CONFIGS.keys()])
    parser.add_argument("--data", default=os.environ.get("YOLO_DATA", str(DEFAULT_DATA)))
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED if DEFAULT_PRETRAINED.exists() else "yolo11n.pt"))
    parser.add_argument("--project", default=None, help="Optional Ultralytics project output directory")
    parser.add_argument("--verify", action="store_true", help="Run module and YAML construction checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        verify_modules()
        return

    modes = DEFAULT_EXPERIMENTS if args.mode == "all" else [args.mode]
    results: dict[str, str] = {}
    for mode in modes:
        try:
            train_model(mode, data=args.data, pretrained=args.pretrained, project=args.project)
            results[mode] = "done"
        except Exception as exc:
            results[mode] = f"failed: {exc}"
            print(f"[error] {mode}: {exc}")
            if args.mode != "all":
                raise

    if args.mode == "all":
        print("\n" + "=" * 70)
        print("v2 ablation summary")
        print("=" * 70)
        for mode, status in results.items():
            print(f"{mode:26s}: {status}")


if __name__ == "__main__":
    main()
