#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vis_from_preds.py
-----------------
Paper-friendly visualization for GazeUnconstrained yaw/pitch predictions.

What it does:
- Reads predictions CSV produced by test_loso.py (preds_{pid}.csv)
- Re-loads the same subject samples in the same order (shuffle=False, augment=False)
- Saves TWO separate images per sample:
    000000_gt.png   (GT arrow only, red)
    000000_pred.png (Pred arrow only, green)
- No header text, no eye crops, no montage (face only)

Key controls:
- --origin_x / --origin_y : arrow origin in face image (ratio). Default near between-eyes.
- --flip_pitch : flip vertical direction (if pitch sign convention differs)
- --flip_yaw   : flip horizontal direction (if yaw sign convention differs)
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gaze_dataset_multi_r50_face_eye_old import GazeDatasetMulti as GazeDataset

try:
    import cv2
except ImportError:
    cv2 = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_cfg_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def unnormalize_to_u8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: CHW torch tensor normalized with ImageNet mean/std
    return: HWC uint8 RGB
    """
    x = img_t.detach().cpu().float().numpy()  # CHW
    x = (x * IMAGENET_STD.reshape(3, 1, 1)) + IMAGENET_MEAN.reshape(3, 1, 1)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return np.transpose(x, (1, 2, 0))  # HWC RGB


def draw_yaw_pitch_arrow(
    img_u8_rgb: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    *,
    color_bgr=(0, 0, 255),
    thickness: int = 3,
    flip_pitch: bool = False,
    flip_yaw: bool = False,
    origin_ratio=(0.50, 0.40),
    length_ratio: float = 0.40,
) -> np.ndarray:
    """
    Draw a single arrow representing yaw/pitch on a face image.

    Mapping:
      dx ~ tan(yaw), dy ~ -tan(pitch) by default (image y-axis downward)
      - flip_pitch toggles dy sign
      - flip_yaw toggles dx sign (equivalently yaw sign)

    origin_ratio:
      (x_ratio, y_ratio) in [0,1] relative to (W, H)
      Default (0.50, 0.40): around between eyes / upper nose bridge for typical face crops.

    length_ratio:
      arrow length relative to min(H, W), fixed-length for clean comparison figures.
    """
    if cv2 is None:
        return img_u8_rgb

    img = img_u8_rgb.copy()
    h, w = img.shape[:2]

    cx = int(w * float(origin_ratio[0]))
    cy = int(h * float(origin_ratio[1]))

    # Apply optional sign flips at the angle level
    yaw_val = -float(yaw_deg) if flip_yaw else float(yaw_deg)
    pitch_val = float(pitch_deg)

    yaw = math.radians(yaw_val)
    pitch = math.radians(pitch_val)

    dx = math.tan(yaw)
    dy = (math.tan(pitch) if flip_pitch else -math.tan(pitch))

    # Normalize direction to get a consistent arrow length
    norm = math.sqrt(dx * dx + dy * dy) + 1e-6
    dx /= norm
    dy /= norm

    L = int(float(length_ratio) * min(h, w))
    x2 = int(cx + dx * L)
    y2 = int(cy + dy * L)

    bgr = img[:, :, ::-1].copy()
    cv2.arrowedLine(bgr, (cx, cy), (x2, y2), color_bgr, int(thickness), tipLength=0.22)
    return bgr[:, :, ::-1]


def load_preds_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "idx": int(row["idx"]),
                "gt_yaw": float(row["gt_yaw"]),
                "gt_pitch": float(row["gt_pitch"]),
                "pred_yaw": float(row["pred_yaw"]),
                "pred_pitch": float(row["pred_pitch"]),
                "abs_err_yaw": float(row.get("abs_err_yaw", "nan")),
                "abs_err_pitch": float(row.get("abs_err_pitch", "nan")),
                "abs_err_mean": float(row.get("abs_err_mean", "nan")),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--pid", required=True, help="e.g., p00")
    ap.add_argument("--pred_csv", required=True, help="e.g., outputs_test/preds_p00.csv")
    ap.add_argument("--out_dir", default="outputs_vis")
    ap.add_argument("--max_samples", type=int, default=200)

    # Visualization controls
    ap.add_argument("--flip_pitch", action="store_true",
                    help="flip pitch direction if arrows look vertically inverted")
    ap.add_argument("--flip_yaw", action="store_true",
                    help="flip yaw direction if arrows look horizontally inverted")
    ap.add_argument("--origin_x", type=float, default=0.50,
                    help="arrow origin x ratio (default 0.50)")
    ap.add_argument("--origin_y", type=float, default=0.26,
                    help="arrow origin y ratio (default 0.40, between-eyes-ish)")
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--length_ratio", type=float, default=0.40)

    args = ap.parse_args()

    if cv2 is None:
        raise RuntimeError("cv2 not installed. Please install: pip install opencv-python")

    cfg = load_cfg_yaml(args.config)

    out_root = Path(args.out_dir)
    vis_dir = out_root / f"vis_{args.pid}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    preds = load_preds_csv(Path(args.pred_csv))
    preds = preds[:args.max_samples]

    # Dataset must match same filtering & ordering, no augment
    ds = GazeDataset(
        cfg["dataset"]["root"], [args.pid],
        cfg["dataset"]["face_size"], cfg["dataset"]["eye_size"],
        augment=False,
        use_frame=cfg.get("dataset", {}).get("use_frame", False),
        hflip_p=0.0
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    origin_ratio = (args.origin_x, args.origin_y)

    for i, bt in enumerate(dl):
        if i >= len(preds):
            break

        r = preds[i]
        face_rgb = unnormalize_to_u8(bt["face"][0])

        # --- Ground Truth image (red) ---
        gt_rgb = draw_yaw_pitch_arrow(
            face_rgb, r["gt_yaw"], r["gt_pitch"],
            color_bgr=(0, 0, 255),
            thickness=args.thickness,
            flip_pitch=args.flip_pitch,
            flip_yaw=args.flip_yaw,
            origin_ratio=origin_ratio,
            length_ratio=args.length_ratio
        )

        # --- Prediction image (green) ---
        pred_rgb = draw_yaw_pitch_arrow(
            face_rgb, r["pred_yaw"], r["pred_pitch"],
            color_bgr=(0, 255, 0),
            thickness=args.thickness,
            flip_pitch=args.flip_pitch,
            flip_yaw=args.flip_yaw,
            origin_ratio=origin_ratio,
            length_ratio=args.length_ratio
        )

        # Save as PNG (cv2 expects BGR)
        cv2.imwrite(str(vis_dir / f"{i:06d}_gt.png"), gt_rgb[:, :, ::-1])
        cv2.imwrite(str(vis_dir / f"{i:06d}_pred.png"), pred_rgb[:, :, ::-1])

    print("[Done]")
    print("Saved to:", vis_dir)
    print("Example:", vis_dir / "000000_gt.png", "and", vis_dir / "000000_pred.png")
    print("Options used:",
          f"flip_pitch={args.flip_pitch}, flip_yaw={args.flip_yaw}, "
          f"origin=({args.origin_x:.2f},{args.origin_y:.2f}), "
          f"thickness={args.thickness}, length_ratio={args.length_ratio}")


if __name__ == "__main__":
    main()
