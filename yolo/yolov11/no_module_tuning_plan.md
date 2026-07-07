# YOLO11n-seg No-Module Tuning Plan

This plan is for the local dataset at `yolo/datasets/severstal-steel-defect-instance-segmentation.v4i.yolov11`.
It assumes we do **not** add custom modules, change the model YAML, or switch away from `yolo11n-seg.pt`.

## Why this plan exists

- The current local `train_4060_balanced4` run peaked near `mAP50(M)=0.43288`.
- The strongest verified baseline so far still comes from the `imgsz=800 + rect=True + SGD` recipe.
- The previous 640-based tuning runs did not beat that baseline, so the next step should isolate changes around the strong 800 setting instead of changing everything at once.

## The three runs

1. `baseline800_rect`
   - Purpose: reproduce the strongest known no-module baseline with the original training shape logic.
   - Main settings: `imgsz=800`, `rect=True`, `optimizer=SGD`, `mask_ratio=4`.

2. `mask2_800_rect`
   - Purpose: keep the strong baseline intact and only test whether finer mask supervision helps small defects.
   - Main changes vs `baseline800_rect`: `mask_ratio=2`.

3. `mask2_800_softaug`
   - Purpose: test whether gentler augmentation helps once the strong 800 baseline and `mask_ratio=2` are both kept.
   - Main changes vs `mask2_800_rect`: lower `mosaic`, later mosaic shutdown, and gentler geometry/color augmentation.

## How to run

Run all three experiments:

```powershell
python yolo/yolov11/main_nomodule_tune.py
```

Each run will save to `yolo/yolov11/runs/segment/<preset>_runs`.

## Recommended order

1. `baseline800_rect`
2. `mask2_800_rect`
3. `mask2_800_softaug`

This order tells us whether the gain comes from preserving the strong 800 baseline, adding finer mask detail, or softening augmentation without changing the network.
