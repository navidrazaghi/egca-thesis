"""Verify that 64 x 2 accumulation is the same update as a true batch of 128.

Gradient accumulation is easy to get subtly wrong and impossible to notice
afterwards: forget to divide the loss by the number of micro-batches and the
summed gradient is `accum` times too large, so the run trains at a learning rate
`accum` times the one written in the config and in Table 5-2.  Nothing errors,
the loss still falls, and the number in the thesis is wrong.

This compares the gradient produced by one backward pass over 128 samples with
the gradient accumulated over two passes of 64 of the same samples in the same
order.  They should agree to floating-point noise.  AMP is off here on purpose:
fp16 rounding would blur exactly the discrepancy this is looking for.

Usage:
    PYTHONPATH=. python tools/check_grad_accum.py --root ~/transfuser/data
"""
import argparse
import os
import sys

import numpy as np
import torch


def grads_of(model):
    return {n: p.grad.detach().float().clone()
            for n, p in model.named_parameters() if p.grad is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="config overrides, to isolate which loss term differs")
    a = ap.parse_args()

    from egca.config import load_config
    from egca.data.transfuser_dataset import TransfuserDataset
    from egca.data.dataset import collate
    from egca.models import EGCAPolicy
    from egca.training.losses import UncertaintyWeightedLoss

    cfg = load_config(a.config, ["model.aux.bev_classes=3"] + a.overrides)
    device = torch.device("cuda")
    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), ["Town01"],
                           augment=False, split="train")

    torch.manual_seed(0)
    model = EGCAPolicy(cfg, sensor_dropout=0.0).to(device)
    crit = UncertaintyWeightedLoss(
        cfg.model.aux.bev_seg, cfg.model.aux.depth,
        wp_dt=cfg.model.decoder.wp_dt,
        speed_bins=getattr(cfg.model.decoder, "speed_bins", None)).to(device)
    model.eval()          # freeze BN so the two paths see identical statistics

    rng = np.random.default_rng(0)
    idx = [int(i) for i in rng.integers(0, len(ds), a.batch)]
    samples = [ds[i] for i in idx]

    def loss_of(sub):
        batch = {k: v.to(device) for k, v in collate(sub).items()}
        return crit(model(batch), batch)[0]

    # one pass over the whole batch
    model.zero_grad(set_to_none=True)
    loss_of(samples).backward()
    whole = grads_of(model)

    # the same samples, in the same order, as `accum` micro-batches
    micro = a.batch // a.accum
    model.zero_grad(set_to_none=True)
    for j in range(a.accum):
        (loss_of(samples[j * micro:(j + 1) * micro]) / a.accum).backward()
    split = grads_of(model)

    # Flattening every parameter into one vector is the right comparison. A
    # per-tensor relative error divides by that tensor's own magnitude, so a
    # layer whose gradient is legitimately near zero reports a huge relative
    # deviation while contributing nothing to the update.
    flat_w = torch.cat([g.flatten() for _, g in sorted(whole.items())])
    flat_s = torch.cat([split[n].flatten() for n, _ in sorted(whole.items())])
    ratio = (flat_s.norm() / flat_w.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(
        flat_s.unsqueeze(0), flat_w.unsqueeze(0)).item()
    rel = ((flat_s - flat_w).norm() / flat_w.norm().clamp_min(1e-12)).item()

    print(f"batch {a.batch} in {a.accum} micro-batches of {micro}")
    print(f"  aux heads: seg={cfg.model.aux.bev_seg} depth={cfg.model.aux.depth}")
    print(f"  parameters compared   {len(whole)}")
    print(f"  gradient norm ratio   {ratio:.6f}   (1.0 correct, "
          f"{a.accum}.0 means the divide was forgotten)")
    print(f"  cosine similarity     {cos:.6f}")
    print(f"  relative L2 deviation {rel:.2e}")
    # The direction has to match; the magnitude has to match to within the
    # rounding two different conv algorithms produce at two different batch
    # sizes, which is far looser than fp32 round-off but far tighter than any
    # real bug in the accumulation.
    ok = cos > 0.9999 and abs(ratio - 1.0) < 0.01
    print("\nPASS" if ok else "\nFAIL: accumulation is not equivalent")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
