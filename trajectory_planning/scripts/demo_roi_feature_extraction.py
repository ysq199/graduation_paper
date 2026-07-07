from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "roi_feature_demo"


def make_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (360, 240), 0)
    draw = ImageDraw.Draw(image)
    blade_like = [(58, 115), (94, 72), (182, 45), (302, 62), (318, 96), (278, 128), (157, 155), (72, 150)]
    draw.polygon(blade_like, fill=255)
    draw.ellipse((245, 92, 322, 157), fill=255)
    draw.rectangle((82, 150, 132, 178), fill=255)
    image.save(path)


def make_point_cloud(path: Path) -> None:
    rng = np.random.default_rng(7)
    u = np.linspace(-1.0, 1.0, 60)
    v = np.linspace(-0.38, 0.38, 24)
    pts = []
    for x in u:
        for y in v:
            z = 0.16 * math.sin(2.2 * x) + 0.09 * y**2 + rng.normal(0.0, 0.003)
            pts.append((x, y, z))
    pts = np.asarray(pts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, pts, fmt="%.7f", delimiter=",")


def main() -> int:
    mask_path = OUT / "demo_roi_mask.png"
    cloud_path = OUT / "demo_surface_cloud.csv"
    json_path = OUT / "demo_features.json"
    contour_csv = OUT / "demo_contour.csv"
    point_csv = OUT / "demo_point_features.csv"

    make_mask(mask_path)
    make_point_cloud(cloud_path)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "extract_roi_features.py"),
        "--image-mask",
        str(mask_path),
        "--point-cloud",
        str(cloud_path),
        "--output-json",
        str(json_path),
        "--output-contour-csv",
        str(contour_csv),
        "--output-point-features-csv",
        str(point_csv),
    ]
    subprocess.check_call(cmd)
    print(json_path)
    print(contour_csv)
    print(point_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
