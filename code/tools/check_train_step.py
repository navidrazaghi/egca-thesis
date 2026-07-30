"""Run every queued configuration through a real training step before the queue.

Building a model and calling forward proves very little about a 9-hour run.  The
things that actually break -- a module that is allocated but never reached, a
loss term that is NaN under mixed precision, a checkpoint that will not reload,
a shape that only fails on real data -- all live past the forward pass.  This
exercises the same path `train.py` takes: autocast, GradScaler, backward,
gradient clipping, optimizer step, save, reload, and an evaluation-mode pass.

Checks per configuration:
  1. two training steps on real data; every loss term finite
  2. the loss actually moves (a frozen loss means nothing is learning)
  3. no parameter is left without gradient -- that is dead weight being carried,
     counted and optimised for nothing
  4. checkpoint round-trips: reload reproduces the same output bit for bit
  5. evaluation mode runs and returns the keys the agent reads

Usage:
    python tools/check_train_step.py --config configs/egca.yaml
"""
import argparse
import copy
import os
import sys
import tempfile

import torch


CONFIGS = {
    "egca_aug (control)": [],
    "egca_query": ["model.decoder.readout=query",
                   "model.fusion.goal_injection=fusion"],
    "egca_regnet": ["model.camera.backbone=regnet_y_3_2gf"],
    "egca_goal": ["model.fusion.goal_injection=fusion"],
}


def pick_device(requested):
    """Prefer the GPU, but do not fail because something else is using it.

    Mixed precision only exists on CUDA, so a CPU run leaves the fp16 paths
    unexercised -- notably the linear attention, whose accumulator sums 4096
    strictly positive terms and is the one place fp16 has already had to be
    worked around.  The summary says plainly which was covered so a CPU pass is
    never mistaken for full clearance.
    """
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(256, 1024, 1024, device="cuda")   # ~1 GB probe
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except RuntimeError:
        print("GPU is busy; falling back to CPU (mixed precision NOT covered)")
        return torch.device("cpu")


def real_batch(cfg, device, n=2):
    """One collated batch of real frames, so shapes come from the dataset."""
    from egca.data.dataset import CarlaDrivingDataset, collate
    ds = CarlaDrivingDataset(cfg, cfg.data.towns_train, augment=True,
                             split="train")
    if len(ds) == 0:
        return None
    batch = collate([ds[i] for i in range(min(n, len(ds)))])
    return {k: v.to(device) for k, v in batch.items()}


def check(name, overrides, base_config, device, batch):
    from egca.config import load_config, Cfg
    from egca.models import EGCAPolicy
    from egca.training.losses import UncertaintyWeightedLoss

    cfg = load_config(base_config, overrides + ["model.camera.pretrained=false"])
    amp = bool(cfg.train.amp) and device.type == "cuda"
    model = EGCAPolicy(cfg, sensor_dropout=cfg.train.sensor_dropout).to(device)
    criterion = UncertaintyWeightedLoss(
        cfg.model.aux.bev_seg, cfg.model.aux.depth,
        wp_dt=cfg.model.decoder.wp_dt,
        speed_bins=getattr(cfg.model.decoder, "speed_bins", None)).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.Adam(params, lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    scaler = torch.GradScaler(enabled=amp)

    ok = True
    print(f"\n=== {name} ===")
    print(f"  {model.num_parameters() / 1e6:.2f} M params, amp={amp}")

    losses = []
    for it in range(2):
        model.train()
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(batch)
            loss, parts = criterion(out, batch)
        bad = [k for k, v in parts.items() if v != v or abs(v) == float("inf")]
        if bad:
            print(f"  FAIL: non-finite loss terms {bad}")
            return False
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if it == 0:
            missing = [n for n, p in model.named_parameters()
                       if p.requires_grad and p.grad is None]
            if missing:
                print(f"  FAIL: {len(missing)} parameters never used, e.g. "
                      + ", ".join(missing[:3]))
                ok = False
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))

    print("  loss terms: " + ", ".join(f"{k}={v:.3f}" for k, v in parts.items()))
    if losses[0] == losses[1]:
        print("  FAIL: loss did not change after an optimiser step")
        ok = False

    # checkpoint round trip, exactly as train.py writes it
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pth")
        torch.save({"model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                    "cfg": dict(cfg), "epoch": 0}, path)
        ck = torch.load(path, map_location=device, weights_only=False)
        clone = EGCAPolicy(Cfg(ck["cfg"]), sensor_dropout=0.0).to(device)
        clone.load_state_dict(ck["model"], strict=True)
        model.eval(); clone.eval()
        with torch.no_grad():
            a = model(batch)["waypoints"]
            b = clone(batch)["waypoints"]
        if not torch.allclose(a, b, atol=1e-5):
            print(f"  FAIL: reloaded model differs, max |d| = "
                  f"{(a - b).abs().max():.2e}")
            ok = False
        else:
            print("  checkpoint reloads and reproduces the same waypoints")

    # evaluation mode must give the agent what it reads
    with torch.no_grad():
        e = model(batch)
    need = {"waypoints", "gate"}
    if cfg.model.decoder.readout == "query":
        need.add("speed_logits")
    missing = need - set(e)
    if missing:
        print(f"  FAIL: eval output missing {sorted(missing)}")
        ok = False
    else:
        print(f"  eval output keys OK: {sorted(e)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda; auto falls back if the GPU is busy")
    a = ap.parse_args()

    from egca.config import load_config
    device = pick_device(a.device)
    cfg = load_config(a.config, [])
    batch = real_batch(cfg, device)
    if batch is None:
        sys.exit("no real frames available; cannot check shapes against data")
    print(f"real batch: image {tuple(batch['image'].shape)}, "
          f"pillars {tuple(batch['pillar_feats'].shape)}, device {device}")

    results = {n: check(n, ov, a.config, device, batch)
               for n, ov in CONFIGS.items()}
    print("\n" + "=" * 46)
    for n, r in results.items():
        print(f"  {n:<22} {'PASS' if r else 'FAIL'}")
    if device.type != "cuda":
        print("  NOTE: ran on CPU, so the mixed-precision path is unverified.")
        print("        Re-run with --device cuda before starting the queue.")
    print("=" * 46)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
