"""Pure hyperparameter tuning runner for local YOLO11n-seg experiments.

This script intentionally avoids custom YAML modules and architecture edits.
It focuses on four practical runs for the local Severstal instance-segmentation
dataset that is stored under ``yolo/datasets`` in this workspace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

YOLOV11_DIR = Path(__file__).resolve().parent
YOLO_ROOT = YOLOV11_DIR.parent
DEFAULT_DATA = YOLO_ROOT / "datasets" / "severstal-steel-defect-instance-segmentation.v4i.yolov11" / "data.yaml"
DEFAULT_PRETRAINED = YOLOV11_DIR / "yolo11n-seg.pt"
DEFAULT_PROJECT = YOLOV11_DIR / "runs" / "segment_nomodule"

BASE_TRAIN_ARGS: dict[str, Any] = {
    "epochs": 200,
    "device": 0,
    "imgsz": 640,
    "rect": False,
    "batch": 0.70,
    "workers": 2,
    "cache": "disk",
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

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline640": {
        "description": "Match the real 640x640 dataset while keeping the previous SGD recipe as the control run.",
        "overrides": {},
    },
    "adamw640": {
        "description": "Check whether the current run is optimizer-limited without changing the rest of the recipe.",
        "overrides": {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
        },
    },
    "mask2_640": {
        "description": "Reduce mask downsampling to preserve more defect boundary detail in the prototype branch.",
        "overrides": {
            "mask_ratio": 2,
        },
    },
    "combo_softaug": {
        "description": "Best-guess no-module stack: AdamW plus mask_ratio=2 plus gentler geometry augmentation.",
        "overrides": {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "mask_ratio": 2,
            "mosaic": 0.5,
            "close_mosaic": 20,
            "degrees": 2.0,
            "translate": 0.03,
            "scale": 0.20,
            "hsv_s": 0.30,
            "hsv_v": 0.20,
        },
    },
}

DEFAULT_EXPERIMENTS = list(EXPERIMENTS.keys())


def merged_train_args(mode: str) -> dict[str, Any]:
    if mode not in EXPERIMENTS:
        raise ValueError(f"Unknown mode: {mode}. Choices: {', '.join(EXPERIMENTS)}")
    train_args = BASE_TRAIN_ARGS.copy()
    train_args.update(EXPERIMENTS[mode]["overrides"])
    return train_args


def build_model(pretrained: str):
    from ultralytics import YOLO

    return YOLO(pretrained)


def print_mode_details(mode: str, data: str, pretrained: str, project: str | None) -> None:
    payload = {
        "mode": mode,
        "description": EXPERIMENTS[mode]["description"],
        "pretrained": pretrained,
        "data": data,
        "project": project,
        "train_args": merged_train_args(mode),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def train_mode(mode: str, data: str, pretrained: str, project: str | None, dry_run: bool) -> None:
    print("\n" + "=" * 72)
    print(f"nomodule tune: {mode}")
    print(f"description  : {EXPERIMENTS[mode]['description']}")
    print("=" * 72)
    print_mode_details(mode, data=data, pretrained=pretrained, project=project)

    if dry_run:
        print("[dry-run] skipped training")
        return

    model = build_model(pretrained)
    train_args = merged_train_args(mode)
    if project:
        train_args["project"] = project

    model.train(
        data=data,
        name=f"nomodule_{mode}",
        **train_args,
    )
    print(f"[done] {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11n-seg no-module tuning runner")
    parser.add_argument("--mode", default="all", choices=["all", *EXPERIMENTS.keys()])
    parser.add_argument("--data", default=os.environ.get("YOLO_DATA", str(DEFAULT_DATA)))
    parser.add_argument(
        "--pretrained",
        default=str(DEFAULT_PRETRAINED if DEFAULT_PRETRAINED.exists() else "yolo11n-seg.pt"),
    )
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arguments without training")
    parser.add_argument("--list", action="store_true", help="Print all experiment descriptions and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for mode in DEFAULT_EXPERIMENTS:
            print(f"{mode}: {EXPERIMENTS[mode]['description']}")
        return

    modes = DEFAULT_EXPERIMENTS if args.mode == "all" else [args.mode]
    results: dict[str, str] = {}
    for mode in modes:
        try:
            train_mode(
                mode,
                data=args.data,
                pretrained=args.pretrained,
                project=args.project,
                dry_run=args.dry_run,
            )
            results[mode] = "ok"
        except Exception as exc:
            results[mode] = f"failed: {exc}"
            print(f"[error] {mode}: {exc}")
            if args.mode != "all":
                raise

    if args.mode == "all":
        print("\n" + "=" * 72)
        print("nomodule tune summary")
        print("=" * 72)
        for mode, status in results.items():
            print(f"{mode:16s}: {status}")


if __name__ == "__main__":
    main()
