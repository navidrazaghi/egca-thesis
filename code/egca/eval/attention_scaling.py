"""Measured scaling of linear vs full cross-attention (supports Sec. 3-4-4).

Section 3-4-4 derives a break-even condition: replacing softmax attention by the
kernel form pays off only when

    N_q N_k  >  (N_q + N_k) d_h / 2

i.e. only above a certain token resolution.  That is a theoretical statement
about operation counts; this script turns it into a measurement, sweeping the
number of LiDAR tokens while keeping the camera tokens fixed and timing both
operators, plus the complete fusion stack of L blocks.

It reports three quantities per point:
  * multiply-accumulate operations, counted analytically;
  * wall-clock latency, which is what a real-time platform actually cares about
    and which does *not* follow the operation count -- softmax kernels are far
    better optimized, so the measured crossover sits at a higher resolution than
    the arithmetic one;
  * peak memory of a training-style forward+backward, where the quadratic term
    is most visible.

Run it on an idle GPU (never while a CARLA collection is using the device, or
both numbers are meaningless):

    python -m egca.eval.attention_scaling --out results/attention_scaling.json
"""
import argparse
import json
import time

import torch

from ..models.attention import FullCrossAttention, LinearCrossAttention
from ..models.fusion import EGCAFusion


def macs_full(nq, nk, d):
    """QK^T and AV, summed over heads: 2 N_q N_k d."""
    return 2.0 * nq * nk * d


def macs_linear(nq, nk, d, heads):
    """S = sum phi(k) v^T and the per-query product: (N_q + N_k) d_h d."""
    return (nq + nk) * (d / heads) * d


def bench(module, nq, nk, d, device, iters=50, warmup=20, amp=False):
    q = torch.randn(1, nq, d, device=device)
    kv = torch.randn(1, nk, d, device=device)
    module = module.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=amp):
                module(q, kv)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast(device_type=device.type, enabled=amp):
                module(q, kv)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
    return ms


def bench_memory(module, nq, nk, d, device, batch=4):
    """Peak memory of a forward+backward, i.e. the training-time cost."""
    if device.type != "cuda":
        return float("nan")
    module = module.to(device).train()
    q = torch.randn(batch, nq, d, device=device, requires_grad=True)
    kv = torch.randn(batch, nk, d, device=device, requires_grad=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    module(q, kv).sum().backward()
    return torch.cuda.max_memory_allocated() / 1e6


def bench_fusion(kind, nc, nl, d, heads, blocks, ffn, device, iters=20, warmup=10):
    fus = EGCAFusion(dim=d, num_blocks=blocks, num_heads=heads, ffn_dim=ffn,
                     attention=kind, gate=True, sensor_dropout=0.0)
    fus = fus.to(device).eval()
    fc = torch.randn(1, nc, d, device=device)
    fl = torch.randn(1, nl, d, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            fus(fc, fl)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fus(fc, fl)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
    del fus, fc, fl
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/attention_scaling.json")
    ap.add_argument("--n-cam", type=int, default=440)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--lidar-tokens", type=int, nargs="*",
                    default=[256, 1024, 4096, 16384])
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU found; latency numbers will not be meaningful")
    d, h = args.dim, args.heads
    d_h = d // h
    rows = []
    print(f"device={device}  N_c={args.n_cam}  d={d}  H={h}  d_h={d_h}\n")
    hdr = (f"{'N_l':>7} {'MAC full':>11} {'MAC lin':>11} {'ratio':>6} "
           f"{'op full':>9} {'op lin':>9} {'fusion full':>12} {'fusion lin':>11} "
           f"{'mem full':>9} {'mem lin':>9}")
    print(hdr)
    print("-" * len(hdr))
    def guard(fn, *a, **kw):
        """Out of memory at a large token count is itself a result, not a crash:
        it is the quadratic term of full attention becoming unaffordable."""
        try:
            return fn(*a, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan")

    for nl in args.lidar_tokens:
        nc = args.n_cam
        mf = macs_full(nc, nl, d)
        ml = macs_linear(nc, nl, d, h)
        t_full = guard(bench, FullCrossAttention(d, h), nc, nl, d, device, amp=args.amp)
        t_lin = guard(bench, LinearCrossAttention(d, h), nc, nl, d, device, amp=args.amp)
        m_full = guard(bench_memory, FullCrossAttention(d, h), nc, nl, d, device)
        m_lin = guard(bench_memory, LinearCrossAttention(d, h), nc, nl, d, device)
        f_full = guard(bench_fusion, "full", nc, nl, d, h, args.blocks, args.ffn, device)
        f_lin = guard(bench_fusion, "linear", nc, nl, d, h, args.blocks, args.ffn, device)
        rows.append({
            "n_cam": nc, "n_lidar": nl, "dim": d, "heads": h,
            "macs_full_G": mf / 1e9, "macs_linear_G": ml / 1e9,
            "mac_ratio": mf / ml,
            "op_full_ms": t_full, "op_linear_ms": t_lin,
            "fusion_full_ms": f_full, "fusion_linear_ms": f_lin,
            "mem_full_MB": m_full, "mem_linear_MB": m_lin,
        })
        print(f"{nl:7d} {mf/1e9:11.3f} {ml/1e9:11.3f} {mf/ml:6.1f} "
              f"{t_full:9.3f} {t_lin:9.3f} {f_full:12.2f} {f_lin:11.2f} "
              f"{m_full:9.0f} {m_lin:9.0f}")

    # the two crossovers: arithmetic (from the condition of Sec. 3-4-4) and the
    # one that actually matters on hardware
    theo = None
    for r in rows:
        if r["mac_ratio"] > 1.0:
            theo = r["n_lidar"]
            break
    meas = None
    for r in rows:
        if r["op_linear_ms"] < r["op_full_ms"]:
            meas = r["n_lidar"]
            break
    print(f"\narithmetic break-even at N_l ~ {theo}, "
          f"measured latency break-even at N_l ~ {meas}")
    print(f"operating point of the thesis: N_c={args.n_cam}, N_l=4096")

    out = {"device": torch.cuda.get_device_name(0) if device.type == "cuda"
           else "cpu",
           "n_cam": args.n_cam, "dim": d, "heads": h, "blocks": args.blocks,
           "amp": bool(args.amp), "rows": rows,
           "breakeven_macs": theo, "breakeven_latency": meas}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
