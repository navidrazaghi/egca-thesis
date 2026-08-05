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
    agg, n, skipped = {}, 0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=cfg.train.amp):
            out = model(batch)
            loss, parts = criterion(out, batch)
        # A single non-finite batch used to sink a whole run in silence: the
        # epoch average went to NaN, the weights absorbed it, and the log said
        # only "train nan" with no indication of how many batches or when. The
        # cause was a dataset defect that the loader now filters, so this should
        # never fire -- which is exactly why it reports rather than just skips.
        if not torch.isfinite(loss):
            skipped += 1
            if skipped <= 3:
                print(f"  batch {i}: non-finite loss, skipped "
                      f"({', '.join(f'{k}={v:.3g}' for k, v in parts.items())})",
                      flush=True)
            continue
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
        if train and i % cfg.train.log_every == 0:
            if writer:
                for k, v in parts.items():
                    writer.add_scalar(f"train/{k}", v, step0 + i)
            # An epoch on this dataset is nearly an hour, so without this the
            # log is silent for the length of a film and a diverging run is
            # only visible once it has burned the hour. The running mean is the
            # epoch's own aggregate, not the batch, so it is comparable to the
            # epoch line that follows.
            rate = (i + 1) * loader.batch_size / max(time.time() - t0, 1e-6)
            print(f"  step {i:5d}/{len(loader)}  "
                  f"loss {agg['total'] / max(n, 1):.4f}  "
                  f"{rate:.0f} samples/s", flush=True)
    if skipped:
        print(f"  {skipped} non-finite batches skipped this pass", flush=True)
    return {k: v / max(n, 1) for k, v in agg.items()}, step0 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="use N random samples instead of real data (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pth in the checkpoint directory")
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
    source = getattr(cfg.data, "source", "own")
    if source == "transfuser":
        # The published dataset the baselines were trained on: 258,866 frames
        # over 488 routes and 8 towns, against 137k over ~50 routes here. Split
        # by town rather than by route -- with eight available, holding two out
        # measures generalisation to unseen layout rather than to unseen frames.
        from ..data.transfuser_dataset import TransfuserDataset
        root = os.path.expanduser(cfg.data.transfuser_root)
        val_towns = list(getattr(cfg.data, "tf_towns_val", ["Town05"]))
        all_towns = list(getattr(cfg.data, "tf_towns_all",
                                 ["Town01", "Town02", "Town03", "Town04",
                                  "Town05", "Town06", "Town07", "Town10HD"]))
        train_towns = [t for t in all_towns if t not in val_towns]
        tr = TransfuserDataset(cfg, root, train_towns, augment=True,
                               split="train")
        va = TransfuserDataset(cfg, root, val_towns, augment=False, split="val")
        # Validation is a monitoring and model-selection signal, and the loader
        # is bound by disk throughput, so scoring all 55,737 Town05 frames every
        # epoch spends a fifth of the run's I/O -- twenty-five times over -- to
        # sharpen a number that is already far inside its own epoch-to-epoch
        # noise. An evenly spaced subset is deterministic, covers every route in
        # the town, and is the same frames every epoch, which is what model
        # selection actually requires.
        cap = int(getattr(cfg.data, "val_max_frames", 0) or 0)
        if 0 < cap < len(va):
            stride = len(va) / cap
            va.frames = [va.frames[int(i * stride)] for i in range(cap)]
            print(f"validation subsampled to {len(va)} of {int(stride * cap)} "
                  f"frames (every ~{stride:.0f}th)")
        print(f"transfuser data: train towns {train_towns}, val {val_towns}")
    else:
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
    # The TransFuser set is 234 GB against 88 GB of page cache, so nearly every
    # read is a cold random one and a worker spends most of a sample blocked on
    # the disk rather than on the CPU: 155 ms per sample, of which about 40 ms
    # is decode. Workers therefore oversubscribe the cores on purpose, and they
    # are kept alive across epochs because respawning 24 of them and refilling
    # their queues costs more than the epoch boundary saves. Threads are pinned
    # to one per worker for the same reason -- 24 workers each opening an
    # OpenCV thread pool is what turned 32 cores into 536 threads before.
    #
    # prefetch_factor stays at the default 2. One batch of 128 is 465 MB, most
    # of it the padded pillar tensors, so the queue holds num_workers * 2 * 465
    # MB: at 24 workers that is 22 GB of the machine's 88, and raising it to 4
    # would take 45 GB out of the page cache that this same disk-bound loader
    # depends on.
    def _pin_threads(_worker_id):
        import cv2
        cv2.setNumThreads(0)
        torch.set_num_threads(1)

    # pin_memory defaults off here, which is the opposite of the usual advice.
    # A batch is 465 MB, and pinned pages cannot be moved or swapped, so several
    # in flight fragment host memory badly enough that kcompactd took 25% of a
    # core, 3 GB of swap filled, and throughput fell from 67 to 43 samples/s
    # over five epochs. Pinning buys faster host-to-device copies, and this
    # loader is bound by random reads off a 234 GB dataset -- the transfer was
    # never the constraint.
    nw = 0 if args.synthetic else cfg.train.num_workers
    pin = bool(getattr(cfg.train, "pin_memory", False))
    extra = dict(persistent_workers=True, prefetch_factor=2,
                 worker_init_fn=_pin_threads) if nw else {}
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate,
                       num_workers=nw, pin_memory=pin, drop_last=True, **extra)
    dl_va = DataLoader(va, batch_size=bs, collate_fn=collate,
                       num_workers=nw, pin_memory=pin, **extra)

    model = EGCAPolicy(cfg, sensor_dropout=cfg.train.sensor_dropout).to(device)
    criterion = UncertaintyWeightedLoss(
        cfg.model.aux.bev_seg, cfg.model.aux.depth,
        wp_dt=cfg.model.decoder.wp_dt,
        speed_bins=getattr(cfg.model.decoder, "speed_bins", None)).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr,
                                 weight_decay=cfg.train.weight_decay)
    scaler = torch.GradScaler(enabled=cfg.train.amp)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.train.ckpt_dir, "tb"))
    print(f"model parameters: {model.num_parameters() / 1e6:.1f} M")

    best, step, start_epoch = float("inf"), 0, 0
    # A run on this dataset is most of a day, and this machine has already lost
    # one to a dead simulator and one to swap thrash. Resuming needs the
    # optimizer's Adam moments and the loss weights' own parameters as well as
    # the model: restoring weights alone restarts the moment estimates from
    # zero, which puts a visible transient in the loss and makes the epoch
    # numbers either side of a restart not comparable.
    resume = os.path.join(cfg.train.ckpt_dir, "last.pth")
    if args.resume and os.path.exists(resume):
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        criterion.load_state_dict(state["criterion"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            best, step = state["best"], state["step"]
            start_epoch = state["epoch"] + 1
            print(f"resumed from epoch {state['epoch']}, best val {best:.4f}")
        else:
            print(f"{resume} predates optimizer checkpointing; "
                  f"starting from scratch rather than resuming without it")

    epochs = cfg.train.epochs if not args.synthetic else 2
    for epoch in range(start_epoch, epochs):
        cosine_warmup(optimizer, epoch, cfg)
        t0 = time.time()
        tr_stats, step = run_epoch(model, criterion, dl_tr, optimizer, scaler,
                                   device, cfg, writer, step, train=True)
        with torch.no_grad():
            va_stats, _ = run_epoch(model, criterion, dl_va, optimizer, scaler,
                                    device, cfg, train=False)
        for k, v in va_stats.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        # The weighted totals of train and val are not comparable (the auxiliary
        # heads only run in training mode), so the raw waypoint L1 -- in metres,
        # the only interpretable convergence number -- is printed alongside.
        print(f"epoch {epoch:03d}  train {tr_stats['total']:.4f} "
              f"(wp {tr_stats['wp']:.3f} m)  val {va_stats['total']:.4f} "
              f"(wp {va_stats['wp']:.3f} m)  lr {optimizer.param_groups[0]['lr']:.2e}"
              f"  ({time.time() - t0:.0f}s)")
        writer.add_scalar("val/wp_metres", va_stats["wp"], epoch)
        # `best` is updated before the checkpoint is built, not after: a resumed
        # run that carried a stale `best` would let the next epoch overwrite
        # best.pth with a worse model than the one already on disk.
        improved = va_stats["total"] < best
        if improved:
            best = va_stats["total"]
        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(),
                "cfg": dict(cfg), "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best": best, "step": step}
        torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "last.pth"))
        if improved:
            torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "best.pth"))


if __name__ == "__main__":
    main()
