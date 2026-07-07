from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_planning.quality_feedback import (  # noqa: E402
    QualityObservation,
    QualityThresholds,
    batch_decide_retake,
    decision_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate retake decisions from image-quality observations.")
    parser.add_argument("--quality-json", type=Path, required=True, help="Input JSON list of quality observations.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON list of retake decisions.")
    parser.add_argument("--min-sharpness", type=float, default=0.45, help="Minimum acceptable sharpness score.")
    parser.add_argument("--max-overexposed-ratio", type=float, default=0.08, help="Maximum acceptable overexposed pixel ratio.")
    parser.add_argument("--min-roi-coverage", type=float, default=0.85, help="Minimum acceptable ROI coverage ratio.")
    parser.add_argument("--max-detector-instability", type=float, default=0.25, help="Maximum acceptable detector instability.")
    parser.add_argument("--max-retake-attempts", type=int, default=3, help="Retake limit before marking a region uninspectable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.quality_json.read_text(encoding="utf-8"))
    observations = [QualityObservation(**item) for item in raw]
    thresholds = QualityThresholds(
        min_sharpness=args.min_sharpness,
        max_overexposed_ratio=args.max_overexposed_ratio,
        min_roi_coverage=args.min_roi_coverage,
        max_detector_instability=args.max_detector_instability,
        max_retake_attempts=args.max_retake_attempts,
    )
    decisions = batch_decide_retake(observations, thresholds)
    payload = [decision_to_dict(decision) for decision in decisions]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"observation_count={len(observations)}")
    print(f"retake_count={sum(1 for decision in decisions if decision.action.value != 'none')}")
    print(f"uninspectable_count={sum(1 for decision in decisions if decision.mark_uninspectable)}")
    print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
