"""Wise-IoU v2 style bbox loss patch for Ultralytics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss
from ultralytics.utils.tal import bbox2dist


def _xyxy_iou_terms(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4
    return iou.clamp(min=0.0, max=1.0), rho2, c2


class WiseBboxLoss(nn.Module):
    """Drop-in replacement for Ultralytics BboxLoss using Wise-IoU v2 style focusing."""

    def __init__(self, reg_max: int = 16, momentum: float = 0.01, gamma: float = 1.9):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.momentum = momentum
        self.gamma = gamma
        self.register_buffer("iou_loss_mean", torch.tensor(1.0))

    def _wise_iou_loss(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor) -> torch.Tensor:
        iou, rho2, c2 = _xyxy_iou_terms(pred_bboxes, target_bboxes)
        iou_loss = 1.0 - iou

        if self.training and iou_loss.numel():
            batch_mean = iou_loss.detach().mean()
            self.iou_loss_mean.mul_(1.0 - self.momentum).add_(batch_mean * self.momentum)

        beta = (iou_loss.detach() / self.iou_loss_mean.clamp(min=1e-6)).clamp(min=0.0)
        focus = beta.pow(self.gamma)
        distance = torch.exp((rho2 / c2).detach())
        return distance * focus * iou_loss

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_iou = (self._wise_iou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum()
        loss_iou = loss_iou / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl


def patch_wise_iou_loss() -> None:
    """Patch Ultralytics so newly created detection losses use WiseBboxLoss."""
    import ultralytics.utils.loss as loss_module

    loss_module.BboxLoss = WiseBboxLoss
