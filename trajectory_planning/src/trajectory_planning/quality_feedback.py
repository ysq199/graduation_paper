from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class RetakeAction(str, Enum):
    NONE = "none"
    ADJUST_LIGHTING = "adjust_lighting"
    ADJUST_DISTANCE = "adjust_distance"
    CHANGE_VIEW_ANGLE = "change_view_angle"
    GENERATE_GAP_VIEWPOINT = "generate_gap_viewpoint"
    MARK_UNINSPECTABLE = "mark_uninspectable"


@dataclass
class QualityThresholds:
    min_sharpness: float = 0.45
    max_overexposed_ratio: float = 0.08
    min_roi_coverage: float = 0.85
    max_detector_instability: float = 0.25
    max_retake_attempts: int = 3


@dataclass
class QualityObservation:
    region_id: str
    sharpness: float
    overexposed_ratio: float
    roi_coverage: float
    detector_instability: float = 0.0
    self_occluded: bool = False
    external_occluded: bool = False
    retake_attempts: int = 0


@dataclass
class RetakeDecision:
    region_id: str
    action: RetakeAction
    reason: str
    should_continue_task: bool = True
    mark_uninspectable: bool = False


def decide_retake(
    observation: QualityObservation,
    thresholds: QualityThresholds | None = None,
) -> RetakeDecision:
    thresholds = thresholds or QualityThresholds()
    if observation.retake_attempts >= thresholds.max_retake_attempts:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.MARK_UNINSPECTABLE,
            reason="retake_attempt_limit_reached",
            mark_uninspectable=True,
        )

    if observation.self_occluded:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.MARK_UNINSPECTABLE,
            reason="self_occluded_from_all_reachable_views",
            mark_uninspectable=True,
        )

    if observation.external_occluded:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.CHANGE_VIEW_ANGLE,
            reason="external_occlusion_detected",
        )

    if observation.overexposed_ratio > thresholds.max_overexposed_ratio:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.ADJUST_LIGHTING,
            reason="overexposed_or_specular_region",
        )

    if observation.sharpness < thresholds.min_sharpness:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.ADJUST_DISTANCE,
            reason="defocus_or_depth_of_field_issue",
        )

    if observation.roi_coverage < thresholds.min_roi_coverage:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.GENERATE_GAP_VIEWPOINT,
            reason="roi_coverage_gap",
        )

    if observation.detector_instability > thresholds.max_detector_instability:
        return RetakeDecision(
            region_id=observation.region_id,
            action=RetakeAction.CHANGE_VIEW_ANGLE,
            reason="detector_confidence_instability",
        )

    return RetakeDecision(
        region_id=observation.region_id,
        action=RetakeAction.NONE,
        reason="quality_satisfied",
    )


def batch_decide_retake(
    observations: list[QualityObservation],
    thresholds: QualityThresholds | None = None,
) -> list[RetakeDecision]:
    return [decide_retake(observation, thresholds) for observation in observations]


def decision_to_dict(decision: RetakeDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload["action"] = decision.action.value
    return payload
