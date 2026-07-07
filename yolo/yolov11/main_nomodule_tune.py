"""Run three 800-based no-module YOLO11n-seg tuning experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

YOLOV11_DIR = Path(__file__).resolve().parent
YOLO_ROOT = YOLOV11_DIR.parent
DEFAULT_DATA = YOLO_ROOT / "datasets" / "severstal-steel-defect-instance-segmentation.v4i.yolov11" / "data.yaml"
DEFAULT_PRETRAINED = YOLOV11_DIR / "yolo11n-seg.pt"
DEFAULT_PROJECT = YOLOV11_DIR / "runs" / "segment"

BASE_TRAIN_ARGS: dict[str, Any] = {
    "epochs": 200,
    "device": 0,
    "imgsz": 800,
    "rect": True,
    "batch": 0.70,
    "workers": 2,
    "cache": "disk",
    "verbose": False,
    "amp": True,
    "patience": 50,
    "save": True,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.10,
    "cos_lr": True,
    "overlap_mask": True,
    "mask_ratio": 4,
    "close_mosaic": 10,
    "hsv_s": 0.40,
    "hsv_v": 0.30,
    "degrees": 3.0,
    "translate": 0.05,
    "scale": 0.30,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

EXPERIMENTS: list[tuple[str, dict[str, Any]]] = [
    ("baseline800_rect", {}),
    (
        "mask2_800_rect",
        {
            "mask_ratio": 2,
        },
    ),
    (
        "mask2_800_softaug",
        {
            "mask_ratio": 2,
            "mosaic": 0.5,
            "close_mosaic": 20,
            "degrees": 2.0,
            "translate": 0.03,
            "scale": 0.20,
            "hsv_s": 0.30,
            "hsv_v": 0.20,
        },
    ),
]


def build_model(pretrained: str):
    from ultralytics import YOLO

    return YOLO(pretrained)


def train_mode(mode: str, overrides: dict[str, Any], data: str, pretrained: str) -> None:
    model = build_model(pretrained)
    train_args = BASE_TRAIN_ARGS.copy()
    train_args.update(overrides)

    print(f"[{mode}] start")
    model.train(
        data=data,
        project=str(DEFAULT_PROJECT),
        name=f"{mode}_runs",
        exist_ok=True,
        **train_args,
    )
    print(f"[done] {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three 800-based YOLO11n-seg no-module tuning experiments")
    parser.add_argument("--data", default=os.environ.get("YOLO_DATA", str(DEFAULT_DATA)))
    parser.add_argument(
        "--pretrained",
        default=str(DEFAULT_PRETRAINED if DEFAULT_PRETRAINED.exists() else "yolo11n-seg.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for mode, overrides in EXPERIMENTS:
        train_mode(mode, overrides, data=args.data, pretrained=args.pretrained)


if __name__ == "__main__":
    main()
