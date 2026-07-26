"""Training entry point (Algorithm 1 of the thesis).

Usage:
    python -m egca.training.train --config configs/egca.yaml
    python -m egca.training.train --config configs/egca.yaml \
        --set model.fusion.attention=full train.sensor_dropout=0.0
    python -m egca.training.train --config configs/egca.yaml --synthetic 256
"""
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import load_config
from ..data.dataset import CarlaDrivingDataset, collate
from ..models import EGCAPolicy
from .losses import UncertaintyWeightedLoss


def cosine_warmup(optimizer, epoch, cfg):
    """Cosine schedule with linear warmup (Table 5-2)."""
    warm, total = cfg.train.warmup_epochs, cfg.train.epochs
    if epoch < warm:
        f = (epoch + 1) / warm
    else:
        f = 0.5 * (1 + math.cos(math.pi * (epoch - warm) / max(1, total - warm)))
    for g in optimizer.param_groups:
        g["lr"] = cfg.train.lr * f


def run_epoch(model, criterion, loader, optimizer, scaler, device, cfg,
              writer=None, step0=0, train=True):
    model.train(train)
    agg, n = {}, 0
    for i, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=cfg.train.amp):
            out = model(batch)
            loss, parts = criterion(out, batch)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        if train and writer and i % cfg.train.log_every == 0:
            for k, v in parts.items():
                writer.add_scalar(f"train/{k}", v, step0 + i)
    return {k: v / max(n, 1) for k, v in agg.items()}, step0 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="use N random samples instead of real data (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the run; the three-seed protocol of Sec. 5-2 "
                         "needs this to vary initialization and data order")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(args.device)
    import random as _random
    import numpy as _np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _np.random.seed(args.seed)
    _random.seed(args.seed)

    # Validation comes from held-out routes of the training towns; the unseen
    # town is never touched here (see CarlaDrivingDataset).
    tr = CarlaDrivingDataset(cfg, cfg.data.towns_train, augment=True,
                             synthetic_len=args.synthetic, split="train")
    va = CarlaDrivingDataset(cfg, cfg.data.towns_train,
                             synthetic_len=max(args.synthetic // 4, 0),
                             split="val")
    if not args.synthetic:
        print(f"train frames {len(tr)}   val frames {len(va)}")
    bs = min(cfg.train.batch_size, len(tr))
    if args.synthetic:
        # smoke test: the configured batch (Table 5-2) assumes 4 x 24 GB GPUs;
        # with N_l = 4096 tokens it does not fit on a single small device.
        bs = min(bs, 4)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate,
                       num_workers=0 if args.synthetic else cfg.train.num_workers,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=bs, collate_fn=collate,
                       num_workers=0 if args.synthetic else cfg.train.num_workers)

    model = EGCAPolicy(cfg, sensor_dropout=cfg.train.sensor_dropout).to(device)
    criterion = UncertaintyWeightedLoss(cfg.model.aux.bev_seg,
                                        cfg.model.aux.depth).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr,
                                 weight_decay=cfg.train.weight_decay)
    scaler = torch.GradScaler(enabled=cfg.train.amp)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.train.ckpt_dir, "tb"))
    print(f"model parameters: {model.num_parameters() / 1e6:.1f} M")

    best, step = float("inf"), 0
    epochs = cfg.train.epochs if not args.synthetic else 2
    for epoch in range(epochs):
        cosine_warmup(optimizer, epoch, cfg)
        t0 = time.time()
        tr_stats, step = run_epoch(model, criterion, dl_tr, optimizer, scaler,
                                   device, cfg, writer, step, train=True)
        with torch.no_grad():
            va_stats, _ = run_epoch(model, criterion, dl_va, optimizer, scaler,
                                    device, cfg, train=False)
        for k, v in va_stats.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"epoch {epoch:03d}  train {tr_stats['total']:.4f}  "
              f"val {va_stats['total']:.4f}  ({time.time() - t0:.0f}s)")
        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(),
                "cfg": dict(cfg), "epoch": epoch}
        torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "last.pth"))
        if va_stats["total"] < best:
            best = va_stats["total"]
            torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "best.pth"))


if __name__ == "__main__":
    main()
