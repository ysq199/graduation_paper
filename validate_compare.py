"""Unified validation for a YOLO segmentation model and a SegFormer checkpoint.

This script compares two trained models on the same validation split with the
same metric definitions:

* foreground mIoU
* foreground Dice
* foreground precision / recall
* pixel accuracy

The YOLO model is evaluated through Ultralytics inference. The SegFormer
checkpoint is loaded with a lightweight MiT-B0 + SegFormer decoder
implementation that matches the provided ``.pth`` structure.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


DEFAULT_DATA = r"E:\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
DEFAULT_YOLO_WEIGHTS = r"E:\graduation_paper - 副本\yolo\yolov11\runs\segment\train_4060_balanced4\weights\best.pt"
DEFAULT_SEGFORMER_WEIGHTS = r"E:\graduation_paper\yolo\yolov11\runs\checkpoints\best.pth"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_yaml(path: Path) -> dict:
    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected YAML content in {path}")
        return data

    # Minimal fallback for simple Ultralytics data.yaml files.
    data: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def resolve_split_paths(data_yaml: Path, split: str) -> list[Path]:
    cfg = load_yaml(data_yaml)
    root_value = cfg.get("path", data_yaml.parent)
    root = Path(str(root_value))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = cfg.get(split)
    if split_value is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")

    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = root / split_path

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (split_path.parent / candidate).resolve()
            images.append(candidate)
        return images

    if split_path.is_dir():
        return sorted([p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

    raise FileNotFoundError(f"Cannot resolve split '{split}' from {data_yaml}: {split_path}")


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts[:idx] + ["labels"] + parts[idx + 1 :]
        return Path(*label_parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def letterbox(image: np.ndarray, new_shape: int = 800, color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (left, top)


def normalize_image_rgb(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    image_rgb = (image_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    image_rgb = np.transpose(image_rgb, (2, 0, 1))
    return torch.from_numpy(image_rgb)


def parse_label_file(label_path: Path, width: int, height: int, num_classes: int) -> np.ndarray:
    mask = np.full((height, width), num_classes, dtype=np.uint8)
    if not label_path.exists():
        return mask

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        parts = [float(x) for x in line.split()]
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls < 0 or cls >= num_classes:
            continue

        coords = parts[1:]
        if len(coords) == 4:
            xc, yc, bw, bh = coords
            x1 = int(round((xc - bw / 2) * width))
            y1 = int(round((yc - bh / 2) * height))
            x2 = int(round((xc + bw / 2) * width))
            y2 = int(round((yc + bh / 2) * height))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = cls
            continue

        if len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= width
            pts[:, 1] *= height
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [pts], int(cls))

    return mask


def resize_mask(mask: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    return cv2.resize(mask, size, interpolation=interpolation)


def infer_num_classes_from_data(data_yaml: Path, default: int = 4) -> int:
    cfg = load_yaml(data_yaml)
    names = cfg.get("names")
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)
    return default


def resolve_torch_device(device_arg: str) -> torch.device:
    value = str(device_arg).strip().lower()
    if value in {"cpu", "-1"} or not torch.cuda.is_available():
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value if ":" in value else "cuda:0")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device("cuda:0")


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.g + self.b


class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, patch_size: int, stride: int):
        super().__init__()
        self.kernel_size = patch_size
        self.stride = stride
        self.padding = patch_size // 2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride, padding=self.padding)
        flat_dim = in_chans * patch_size * patch_size
        self.weight = nn.Parameter(torch.empty(out_chans, flat_dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_chans))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.unfold(x)
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        x = x.view(b, x.shape[1], h_out, w_out)
        return F.conv2d(x, self.weight, self.bias)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            DWConv(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        kh, kw = k.shape[-2:]
        k = k.reshape(b, self.num_heads, self.head_dim, kh * kw).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, kh * kw).transpose(2, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(2, 3).reshape(b, c, h, w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(self.norm(x))


class Block(nn.Sequential):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, mlp_ratio: int):
        hidden_dim = dim * mlp_ratio
        super().__init__(
            PreNorm(dim, EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)),
            PreNorm(dim, MixFFN(dim, hidden_dim=hidden_dim)),
        )


class Stage(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, depth: int, num_heads: int, sr_ratio: int, mlp_ratio: int, patch_size: int, stride: int):
        super().__init__()
        self.add_module("0", nn.Identity())
        self.add_module("1", PatchEmbed(in_chans, out_chans, patch_size=patch_size, stride=stride))
        self.add_module("2", nn.ModuleList([Block(out_chans, num_heads=num_heads, sr_ratio=sr_ratio, mlp_ratio=mlp_ratio) for _ in range(depth)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._modules["1"](x)
        for block in self._modules["2"]:
            x = block(x)
        return x


class MiTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                Stage(3, 32, depth=2, num_heads=1, sr_ratio=8, mlp_ratio=8, patch_size=7, stride=4),
                Stage(32, 64, depth=2, num_heads=2, sr_ratio=4, mlp_ratio=8, patch_size=3, stride=2),
                Stage(64, 160, depth=2, num_heads=5, sr_ratio=2, mlp_ratio=4, patch_size=3, stride=2),
                Stage(160, 256, depth=2, num_heads=8, sr_ratio=1, mlp_ratio=4, patch_size=3, stride=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


class SegFormerHead(nn.Module):
    def __init__(self, in_channels: list[int], embed_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.to_fused = nn.ModuleList([nn.Sequential(nn.Conv2d(c, embed_dim, kernel_size=1, bias=True)) for c in in_channels])
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = feats[0].shape[-2:]
        fused = []
        for proj, feat in zip(self.to_fused, feats):
            x = proj(feat)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            fused.append(x)
        x = torch.cat(fused, dim=1)
        x = self.to_segmentation(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class SegFormerB0(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.mit = MiTBackbone()
        self.to_fused = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 256, kernel_size=1, bias=True)) for c in [32, 64, 160, 256]])
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1, bias=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        feats = self.mit(x)
        target_h, target_w = feats[0].shape[-2:]
        fused = []
        for proj, feat in zip(self.to_fused, feats):
            y = proj(feat)
            if y.shape[-2:] != (target_h, target_w):
                y = F.interpolate(y, size=(target_h, target_w), mode="bilinear", align_corners=False)
            fused.append(y)
        y = torch.cat(fused, dim=1)
        y = self.to_segmentation(y)
        if y.shape[-2:] != out_size:
            y = F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)
        return y


def load_segformer_checkpoint(weights: Path, device: torch.device, num_classes: int) -> SegFormerB0:
    ckpt = torch.load(weights, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model = SegFormerB0(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[segformer] load_state_dict warnings:")
        print("  missing   :", missing[:20], "..." if len(missing) > 20 else "")
        print("  unexpected:", unexpected[:20], "..." if len(unexpected) > 20 else "")
    model.to(device).eval()
    return model


@dataclass
class MetricsAccumulator:
    num_classes: int
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    correct: int = 0
    total: int = 0

    @classmethod
    def create(cls, num_classes: int) -> "MetricsAccumulator":
        zeros = np.zeros(num_classes, dtype=np.int64)
        return cls(num_classes=num_classes, tp=zeros.copy(), fp=zeros.copy(), fn=zeros.copy())

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        self.correct += int((pred == target).sum())
        self.total += int(pred.size)
        for c in range(self.num_classes):
            pred_c = pred == c
            tgt_c = target == c
            self.tp[c] += int(np.logical_and(pred_c, tgt_c).sum())
            self.fp[c] += int(np.logical_and(pred_c, ~tgt_c).sum())
            self.fn[c] += int(np.logical_and(~pred_c, tgt_c).sum())

    def summary(self) -> dict[str, float | list[float]]:
        precision = self.tp / np.maximum(self.tp + self.fp, 1)
        recall = self.tp / np.maximum(self.tp + self.fn, 1)
        dice = 2 * self.tp / np.maximum(2 * self.tp + self.fp + self.fn, 1)
        iou = self.tp / np.maximum(self.tp + self.fp + self.fn, 1)
        valid = (self.tp + self.fp + self.fn) > 0

        return {
            "pixel_acc": float(self.correct / max(self.total, 1)),
            "precision_fg": float(np.mean(precision[valid])) if np.any(valid) else 0.0,
            "recall_fg": float(np.mean(recall[valid])) if np.any(valid) else 0.0,
            "dice_fg": float(np.mean(dice[valid])) if np.any(valid) else 0.0,
            "miou_fg": float(np.mean(iou[valid])) if np.any(valid) else 0.0,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "dice_per_class": dice.tolist(),
            "iou_per_class": iou.tolist(),
        }


def semantic_from_yolo_result(result, num_classes: int, orig_shape: tuple[int, int], conf_thresh: float) -> np.ndarray:
    h, w = orig_shape
    score_map = np.zeros((num_classes, h, w), dtype=np.float32)

    if result.masks is None or result.boxes is None:
        return np.full((h, w), num_classes, dtype=np.uint8)

    masks = result.masks.data
    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    for mask, cls_id, conf in zip(masks, cls_ids, confs):
        if cls_id < 0 or cls_id >= num_classes or conf < conf_thresh:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        active = mask > 0.5
        score_map[cls_id][active] = np.maximum(score_map[cls_id][active], conf)

    bg_score = np.zeros((1, h, w), dtype=np.float32)
    stacked = np.concatenate([score_map, bg_score], axis=0)
    return stacked.argmax(axis=0).astype(np.uint8)


def semantic_from_segformer_logits(logits: torch.Tensor, num_classes: int, thresh: float) -> np.ndarray:
    probs = logits.sigmoid().squeeze(0)
    max_prob, cls_map = probs.max(dim=0)
    pred = torch.full_like(cls_map, fill_value=num_classes, dtype=torch.long)
    active = max_prob >= thresh
    pred[active] = cls_map[active]
    return pred.detach().cpu().numpy().astype(np.uint8)


def evaluate_yolo(
    weights: Path,
    image_paths: Iterable[Path],
    num_classes: int,
    imgsz: int,
    device: str,
    conf_thresh: float,
    iou_thresh: float,
) -> tuple[dict[str, float | list[float]], float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    meter = MetricsAccumulator.create(num_classes)
    total_time = 0.0
    count = 0

    for img_path in image_paths:
        image = read_image_bgr(img_path)
        h, w = image.shape[:2]
        label_path = image_to_label_path(img_path)
        target = parse_label_file(label_path, w, h, num_classes)

        t0 = time.perf_counter()
        result = model.predict(
            source=image,
            imgsz=imgsz,
            device=device,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
            retina_masks=True,
        )[0]
        total_time += time.perf_counter() - t0
        count += 1

        pred = semantic_from_yolo_result(result, num_classes=num_classes, orig_shape=(h, w), conf_thresh=conf_thresh)
        meter.update(pred, target)

    summary = meter.summary()
    summary["model"] = "yolo"
    return summary, total_time / max(count, 1)


def evaluate_segformer(
    weights: Path,
    image_paths: Iterable[Path],
    num_classes: int,
    device: torch.device,
    thresh: float,
) -> tuple[dict[str, float | list[float]], float]:
    model = load_segformer_checkpoint(weights, device=device, num_classes=num_classes)
    meter = MetricsAccumulator.create(num_classes)
    total_time = 0.0
    count = 0

    for img_path in image_paths:
        image = read_image_bgr(img_path)
        h, w = image.shape[:2]
        label_path = image_to_label_path(img_path)
        target = parse_label_file(label_path, w, h, num_classes)

        inp = normalize_image_rgb(image).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(inp)
        total_time += time.perf_counter() - t0
        count += 1

        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = semantic_from_segformer_logits(logits, num_classes=num_classes, thresh=thresh)
        meter.update(pred, target)

    summary = meter.summary()
    summary["model"] = "segformer"
    return summary, total_time / max(count, 1)


def print_summary(name: str, summary: dict[str, float | list[float]], avg_time: float) -> None:
    fps = 1.0 / avg_time if avg_time > 0 else math.inf
    print(f"\n{name}")
    print("-" * len(name))
    print(f"pixel_acc : {summary['pixel_acc']:.4f}")
    print(f"miou_fg   : {summary['miou_fg']:.4f}")
    print(f"dice_fg   : {summary['dice_fg']:.4f}")
    print(f"prec_fg   : {summary['precision_fg']:.4f}")
    print(f"recall_fg : {summary['recall_fg']:.4f}")
    print(f"avg_time  : {avg_time:.4f} s")
    print(f"fps       : {fps:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified validation for YOLO and SegFormer")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Ultralytics data.yaml path")
    parser.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS, help="YOLO segmentation weights")
    parser.add_argument("--segformer-weights", default=DEFAULT_SEGFORMER_WEIGHTS, help="SegFormer checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image size")
    parser.add_argument("--device", default="0", help="Ultralytics / torch device, e.g. 0 or cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.001, help="YOLO confidence threshold for semantic fusion")
    parser.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--segformer-thresh", type=float, default=0.5, help="SegFormer per-class probability threshold")
    parser.add_argument("--num-classes", type=int, default=4, help="Foreground class count")
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for a quick dry run")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    image_paths = resolve_split_paths(data_yaml, args.split)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    num_classes = args.num_classes or infer_num_classes_from_data(data_yaml, default=4)
    device = resolve_torch_device(args.device)

    print(f"data       : {data_yaml}")
    print(f"split      : {args.split}")
    print(f"images     : {len(image_paths)}")
    print(f"classes    : {num_classes}")
    print(f"imgsz      : {args.imgsz}")

    yolo_summary, yolo_avg_time = evaluate_yolo(
        weights=Path(args.yolo_weights),
        image_paths=image_paths,
        num_classes=num_classes,
        imgsz=args.imgsz,
        device=args.device,
        conf_thresh=args.yolo_conf,
        iou_thresh=args.yolo_iou,
    )
    print_summary("YOLO", yolo_summary, yolo_avg_time)

    segformer_summary, segformer_avg_time = evaluate_segformer(
        weights=Path(args.segformer_weights),
        image_paths=image_paths,
        num_classes=num_classes,
        device=device,
        thresh=args.segformer_thresh,
    )
    print_summary("SegFormer", segformer_summary, segformer_avg_time)

    result = {
        "config": {
            "data": str(data_yaml),
            "split": args.split,
            "imgsz": args.imgsz,
            "num_classes": num_classes,
            "yolo_weights": str(args.yolo_weights),
            "segformer_weights": str(args.segformer_weights),
        },
        "yolo": {**yolo_summary, "avg_time": yolo_avg_time, "fps": (1.0 / yolo_avg_time if yolo_avg_time > 0 else math.inf)},
        "segformer": {
            **segformer_summary,
            "avg_time": segformer_avg_time,
            "fps": (1.0 / segformer_avg_time if segformer_avg_time > 0 else math.inf),
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
