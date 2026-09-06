#!/usr/bin/env python
# train_loso_multi_freeze_face_eye_r50.py  (ResNet-50 + L1 loss)
# ------------------------------------------------------------
# ❶ 冻结 CNN (BN=eval) 仅训头，差分学习率（head 1e-3 / backbone 1e-4）
# ❷ 指定 epoch 解冻 CNN，重建优化器（仍然差分 LR）
# ❸ 训练/验证都使用 L1 损失（单位：度）
# ❹ 每个被试独立 .log 文件（保存到 save_dir）
# ❺ 可选加载本地预训练并打印 "loaded N tensors ..."
# ------------------------------------------------------------

import argparse, yaml, time, json, math, random, logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaze_dataset_multi_r50_face_eye_old import GazeDatasetMulti as GazeDataset   # 你现有的数据集脚本
from model_multi_r50_face_eye_old import ModelMulti as Model             # 本次提供的 ResNet-50 模型

# -------------------- 基础工具 -------------------- #
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_cfg(path):
    with open(path) as f: 
        return yaml.safe_load(f)

def build_logger(out_file):
    """
    out_file: 完整路径，如 /path/to/save_dir/train_val-p00.log
    作用：把终端输出同时写入 out_file，并避免重复 handler。
    """
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lg = logging.getLogger("train")
    lg.setLevel(logging.INFO)

    # 清理旧 handler（防止多次循环时重复写入同样内容）
    if lg.handlers:
        for h in list(lg.handlers):
            lg.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    fh = logging.FileHandler(out_file, mode="a")
    sh.setFormatter(fmt); fh.setFormatter(fmt)
    lg.addHandler(sh); lg.addHandler(fh)

    logging.captureWarnings(True)
    return lg

# ----------------- 冻结 & 优化器 ------------------ #
def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

def freeze_cnn(model, flag: bool):
    for name in ("back_face", "back_eye", "back_frame"):
        br = getattr(model, name, None)
        if br is None:
            continue
        for p in br.parameters():
            p.requires_grad = not flag
        if flag:
            br.apply(set_bn_eval)

def group_params(model):
    head, back = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(("back_face", "back_eye", "back_frame")):
            back.append(p)
        else:
            head.append(p)
    return head, back

def build_opt(model, wd=1e-4, head_lr=1e-3, bb_lr=1e-4):
    head, back = group_params(model)
    return torch.optim.AdamW(
        [{"params": head, "lr": head_lr, "weight_decay": wd},
         {"params": back, "lr": bb_lr,  "weight_decay": wd}]
    )

def build_scheduler(optimizer, max_epoch, warmup=3):
    import math
    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / max(1, warmup)
        t = (e - warmup) / max(1, (max_epoch - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))  # cos 衰减到 0.1x
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------- 可选：加载本地预训练 ---------------- #
def load_local_pretrain(model, path, logger):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    # 只取 base_model.* 前缀（按你以前习惯），拷到各 CNN 分支
    cnn_state = {k.replace("base_model.", ""): v
                 for k, v in state.items()
                 if k.startswith("base_model.")}

    total = 0
    for name in ("back_face", "back_eye", "back_frame"):
        br = getattr(model, name, None)
        if br is None:
            continue
        own = br.state_dict()
        matched = 0
        for k, v in cnn_state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); matched += 1
        br.load_state_dict(own)
        total += matched
        logger.info(f"[Info] loaded {matched} tensors into {name} ({br._get_name()})")
    if total == 0:
        logger.warning("[Warn] no tensors matched; check maps/channels or key prefix]")

# ---------------- DataLoader ------------------ #
def make_loader(cfg, ids, val=False):
    ds = GazeDataset(
        cfg["dataset"]["root"], ids,
        cfg["dataset"]["face_size"], cfg["dataset"]["eye_size"],
        augment=not val,
        use_frame=cfg.get("model", {}).get("use_frame", False),
        hflip_p=cfg.get("dataset", {}).get("hflip_p", 0.0) if not val else 0.0
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=not val,
        num_workers=cfg["train"]["workers"],
        pin_memory=True,
        drop_last=False
    )

# ---------------- Epoch 循环 ----------------------- #
def run_epoch(model, loader, device, opt=None, train=True, tag="train"):
    model.train() if train else model.eval()
    total = 0.0
    count = 0

    for bt in tqdm(loader, desc=tag, ncols=110, leave=False):
        x = {k: v.to(device) for k, v in bt.items() if k != "label"}
        y = bt["label"].to(device)

        if train:
            pred = model(x)
            assert pred.size(0) == y.size(0), f"Batch mismatch: predB={pred.size(0)}, yB={y.size(0)}"
            loss = model.loss_op(pred, y)       # L1 (degree)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
        else:
            with torch.no_grad():
                pred = model(x)
                assert pred.size(0) == y.size(0), f"Batch mismatch: predB={pred.size(0)}, yB={y.size(0)}"
                loss = F.l1_loss(pred, y, reduction="mean")
                total += loss.item() * y.size(0)
                count += y.size(0)

    return total / max(1, count)

# ---------------- Main ---------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_cfg(args.config); set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)

    root = Path(cfg["dataset"]["root"])
    pids = sorted(p.name for p in root.iterdir() if p.is_dir())

    total_ep  = cfg["train"]["epochs"]
    freeze_ep = cfg["train"]["freeze_epochs"]
    head_lr   = cfg["train"].get("head_lr", 1e-3)
    bb_lr     = cfg["train"].get("backbone_lr", 1e-4)
    weight_decay = cfg["train"]["weight_decay"]
    warmup_ep = cfg["train"].get("warmup_epochs", 3)

    summary = {}

    for val_pid in pids:
        # 为该被试创建独立 .log 文件
        logger = build_logger(str(save_dir / f"train_val-{val_pid}.log"))
        tr_ids = [p for p in pids if p != val_pid]
        logger.info(f"=== LOSO | val {val_pid} | train {len(tr_ids)} subj ===")

        dl_tr = make_loader(cfg, tr_ids, val=False)
        dl_va = make_loader(cfg, [val_pid], val=True)

        model = Model(
            cfg["dataset"]["face_size"],
            cfg["dataset"]["eye_size"],
            maps=cfg.get("model", {}).get("maps", 64),
            layers=cfg.get("model", {}).get("layers", 6),
            max_seq_hint=cfg.get("model", {}).get("max_seq_hint", 1024),
            use_frame=cfg.get("model", {}).get("use_frame", False)
        ).to(device)

        # （可选）加载你自己的预训练，并打印匹配到的 tensor 数
        if cfg.get("pretrain", {}).get("enable", False):
            load_local_pretrain(model, cfg["pretrain"]["path"], logger)

        # 冻结 CNN & 优化器 & 调度
        freeze_cnn(model, True)
        opt = build_opt(model, wd=weight_decay, head_lr=head_lr, bb_lr=bb_lr)
        sch = build_scheduler(opt, max_epoch=total_ep, warmup=warmup_ep)

        best = math.inf
        for ep in range(1, total_ep + 1):
            t0 = time.time()
            tr_mae = run_epoch(model, dl_tr, device, opt, True, tag="train")  # L1 (度)
            va_mae = run_epoch(model, dl_va, device, None, False, tag="val")   # L1 (度)
            sch.step()

            logger.info(
                f"PID {val_pid} | Ep {ep:02d}/{total_ep} | "
                f"Train {tr_mae:.4f} | Val {va_mae:.2f}° | "
                f"LR_head {opt.param_groups[0]['lr']:.1e} | LR_bb {opt.param_groups[1]['lr']:.1e} | "
                f"{time.time()-t0:.1f}s"
            )

            # 指定 epoch 解冻（重建 optimizer，保持差分 LR）
            if ep == freeze_ep:
                logger.info(f"[Info] unfreeze CNN at epoch {ep}")
                freeze_cnn(model, False)
                opt = build_opt(model, wd=weight_decay, head_lr=head_lr, bb_lr=bb_lr)
                sch = build_scheduler(opt, max_epoch=total_ep, warmup=warmup_ep)

            if va_mae < best:
                best = va_mae
                torch.save(model.state_dict(), save_dir / f"best_{val_pid}.pth")

        summary[val_pid] = best
        logger.info(f">>> Best for {val_pid}: {best:.2f}°")

    mean_err = sum(summary.values()) / max(1, len(summary))
    # 总日志
    main_logger = build_logger(str(save_dir / "train_all.log"))
    main_logger.info("=== LOSO Summary ===")
    for pid, err in summary.items():
        main_logger.info(f"{pid}: {err:.2f}°")
    main_logger.info(f"Average Angular Error: {mean_err:.2f}°")

    with (save_dir / "loso_results.json").open("w") as f:
        json.dump({**summary, "mean": mean_err}, f, indent=2)

if __name__ == "__main__":
    main()
