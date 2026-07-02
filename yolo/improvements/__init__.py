"""Custom YOLO improvement modules and Ultralytics registration helpers."""

from .c3k2_lfe import C3k2_LFE, LFE, LFELite
from .dysample import DySample
from .registry import register_improvements
from .simam import C2fSimAM, SimAM
from .slimneck import GSConv, GSBottleneck, VoVGSCSP
from .wiou import WiseBboxLoss, patch_wise_iou_loss

__all__ = [
    "C2fSimAM",
    "C3k2_LFE",
    "DySample",
    "GSBottleneck",
    "GSConv",
    "LFE",
    "LFELite",
    "SimAM",
    "VoVGSCSP",
    "WiseBboxLoss",
    "patch_wise_iou_loss",
    "register_improvements",
]
