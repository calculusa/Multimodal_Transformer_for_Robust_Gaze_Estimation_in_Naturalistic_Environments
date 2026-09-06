#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gaze_dataset_multi_r50_face_eye_old import GazeDatasetMulti as GazeDataset
from model_multi_r50_face_eye_old import ModelMulti as Model





def load_cfg_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pid", required=True, help="e.g. p00")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=None, help="override config train.batch_size if set")
    ap.add_argument("--max_samples", type=int, default=-1, help="-1 means all")
    ap.add_argument("--out_dir", default="outputs_test")
    ap.add_argument("--ckpt_dir", default=None, help="override config train.save_dir if set")
    ap.add_argument("--debug_once", action="store_true", help="print one-batch sanity info")
    args = ap.parse_args()

    cfg = load_cfg_yaml(args.config)

    # ckpt dir: prefer CLI override, else config train.save_dir
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(cfg["train"]["save_dir"])
    ckpt_path = ckpt_dir / f"best_{args.pid}.pth"

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Tip: check train.save_dir in config, or pass --ckpt_dir explicitly."
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset: val mode (no augment, no flip)
    ds = GazeDataset(
        cfg["dataset"]["root"], [args.pid],
        cfg["dataset"]["face_size"], cfg["dataset"]["eye_size"],
        augment=False,
        use_frame=cfg.get("dataset", {}).get("use_frame", False),
        hflip_p=0.0
    )
    bs = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=cfg["train"]["workers"],
        pin_memory=True,
        drop_last=False
    )

    # model (must match training)
    model = Model(
        cfg["dataset"]["face_size"],
        cfg["dataset"]["eye_size"],
        maps=cfg.get("model", {}).get("maps", 64),
        layers=cfg.get("model", {}).get("layers", 6),
        max_seq_hint=cfg.get("model", {}).get("max_seq_hint", 1024),
        use_frame=cfg.get("dataset", {}).get("use_frame", False)
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    print("[INFO] Using config:", args.config)
    print("[INFO] Loading checkpoint:", ckpt_path)

    rows = []
    abs_err_yaw_list, abs_err_pitch_list, abs_err_mean_list = [], [], []

    seen = 0
    for bt in dl:
        y = bt["label"].to(device)  # [B,2] yaw,pitch (deg)
        x = {k: v.to(device) for k, v in bt.items() if k != "label"}

        if args.debug_once and seen == 0:
            print("[DEBUG] keys:", list(x.keys()))
            print("[DEBUG] shapes:",
                  "face", tuple(x["face"].shape),
                  "eye_l", tuple(x["eye_l"].shape),
                  "eye_r", tuple(x["eye_r"].shape))
            if "frame" in x:
                print("[DEBUG] frame", tuple(x["frame"].shape))
            # token S check (exact for your expected strides: face 14x14, eye 4x4)
            # This is a sanity hint; actual depends on your resnet implementation.
            face_S = 14 * 14
            eye_S = 4 * 4
            S_est = face_S + 2 * eye_S + (face_S if ("frame" in x) else 0)
            print(f"[DEBUG] est token S={S_est}, need pos_embed >= {S_est+1}, have {model.pos_embed.num_embeddings}")

        pred = model(x)  # [B,2]
        if pred.ndim != 2 or pred.size(-1) != 2:
            raise ValueError(f"Expected pred [B,2], got {pred.shape}")

        abs_err = (pred - y).abs()

        bsz = pred.size(0)
        for i in range(bsz):
            if args.max_samples > 0 and seen >= args.max_samples:
                break

            gt_yaw, gt_pitch = float(y[i, 0].item()), float(y[i, 1].item())
            pr_yaw, pr_pitch = float(pred[i, 0].item()), float(pred[i, 1].item())

            ey = float(abs_err[i, 0].item())
            ep = float(abs_err[i, 1].item())
            em = float((ey + ep) / 2.0)

            rows.append({
                "idx": seen,
                "gt_yaw": gt_yaw, "gt_pitch": gt_pitch,
                "pred_yaw": pr_yaw, "pred_pitch": pr_pitch,
                "abs_err_yaw": ey, "abs_err_pitch": ep,
                "abs_err_mean": em,
            })

            abs_err_yaw_list.append(ey)
            abs_err_pitch_list.append(ep)
            abs_err_mean_list.append(em)
            seen += 1

        if args.max_samples > 0 and seen >= args.max_samples:
            break

    metrics = {
        "pid": args.pid,
        "n_samples": int(seen),
        "mae_yaw_deg": float(np.mean(abs_err_yaw_list)) if abs_err_yaw_list else float("nan"),
        "mae_pitch_deg": float(np.mean(abs_err_pitch_list)) if abs_err_pitch_list else float("nan"),
        "mae_mean_deg": float(np.mean(abs_err_mean_list)) if abs_err_mean_list else float("nan"),
        "ckpt": str(ckpt_path),
        "config": str(Path(args.config).resolve()),
        "note": "abs_err_mean = (|yaw_err| + |pitch_err|)/2 per sample, then averaged."
    }

    csv_path = out_dir / f"preds_{args.pid}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    json_path = out_dir / f"metrics_{args.pid}.json"
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)



    print("[INFO] Using config:", args.config)
    print("[INFO] Loading checkpoint:", ckpt_path)
    print("[INFO] resnet50 from:", model.back_face.__class__.__module__)


    print("[Done]")
    print(" CSV  :", csv_path)
    print(" JSON :", json_path)


if __name__ == "__main__":
    main()
