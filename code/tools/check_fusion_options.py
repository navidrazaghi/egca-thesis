"""Check the new fusion options without breaking anything that already exists.

Two things have to hold at once.  The pooling and gate options added to address
the saturated gate must actually change the model, and every checkpoint trained
before they existed must still build and load byte-for-byte -- `evaluate` and
the leaderboard agent both reconstruct the architecture from the config stored
inside the checkpoint, so a silent default change would break eleven trained
runs at once.

Usage:
    python tools/check_fusion_options.py --config configs/egca.yaml \
        [--checkpoints checkpoints]
"""
import argparse
import glob
import os
import sys

import torch


def build(path, **overrides):
    """Rebuild from the YAML with dotted overrides.

    Not setattr on a loaded config: Cfg.__getattr__ wraps each nested dict in a
    fresh Cfg on every access, so assigning to `cfg.model.fusion.pooling` writes
    to a temporary and is silently lost -- which is exactly how the first run of
    this check reported "option changed nothing" for all three variants.
    """
    from egca.config import load_config
    from egca.models import EGCAPolicy
    ovs = [f"model.fusion.{k}={v}" for k, v in overrides.items()]
    return EGCAPolicy(load_config(path, ovs), sensor_dropout=0.0).eval()


def synthetic_batch(cfg, b=2):
    d = cfg.model.lidar
    n_pillars, m = 64, d.max_points_per_pillar
    h, w = cfg.model.camera.image_size
    return {
        "image": torch.randn(b, 3, h, w),
        "pillar_feats": torch.randn(b * n_pillars, m, 9),
        "pillar_coords": torch.randint(0, 100, (b * n_pillars, 2)),
        "pillar_mask": torch.ones(b * n_pillars, m),
        "pillar_batch": torch.arange(b).repeat_interleave(n_pillars),
        "speed": torch.rand(b, 1) * 8,
        "command": torch.randint(0, 4, (b,)),
        "goal": torch.randn(b, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    ap.add_argument("--checkpoints", default="checkpoints")
    a = ap.parse_args()

    from egca.config import load_config
    cfg = load_config(a.config, [])
    ok = True

    # ---- 1. defaults must still be the original architecture ---------------
    base = build(a.config, pooling="mean", gate_form="scalar")
    n_base = base.num_parameters()
    print(f"baseline (mean pooling, scalar gate): {n_base / 1e6:.2f} M params")

    with torch.no_grad():
        out = base(synthetic_batch(cfg))
    T = cfg.model.decoder.horizon
    assert out["waypoints"].shape == (2, T, 2), out["waypoints"].shape
    assert out["gate"].shape == (2,), out["gate"].shape
    print("   forward OK, gate is one number per sample")

    # ---- 2. the new options must build and change the model ----------------
    for pooling, gate_form, space, goal in (
            ("attention", "vector", "image", "decoder"),
            ("mean", "scalar", "bev", "decoder"),
            ("mean", "scalar", "image", "fusion")):
        mdl = build(a.config, pooling=pooling, gate_form=gate_form,
                    camera_space=space, goal_injection=goal)
        n = mdl.num_parameters()
        with torch.no_grad():
            o = mdl(synthetic_batch(cfg))
        assert o["waypoints"].shape == (2, T, 2)
        assert o["gate"].shape == (2,), f"{pooling}/{gate_form}: {o['gate'].shape}"
        delta = (n - n_base) / 1e6
        print(f"{pooling:>9} pool, {gate_form:>6} gate, {space:>5} space, "
              f"goal {goal:<7}: {n / 1e6:.2f} M ({delta:+.2f} M), OK")
        if n == n_base:
            print("   FAIL: option changed nothing")
            ok = False

    # ---- 2a. the query readout: shapes, speed head and its target ----------
    from egca.config import load_config as _lc
    from egca.models import EGCAPolicy as _P
    # A query readout with the goal left in the decoder has no path for the goal
    # at all; the model must refuse to build rather than drive blind.
    try:
        _P(_lc(a.config, ["model.decoder.readout=query"]), sensor_dropout=0.0)
        print("   FAIL: query readout built with no route for the goal")
        ok = False
    except ValueError:
        print("query readout rejects a goal with nowhere to go")

    qm = _P(_lc(a.config, ["model.decoder.readout=query",
                           "model.fusion.goal_injection=fusion"]),
            sensor_dropout=0.0)
    qm.train()                      # auxiliary heads and speed head are training-only
    qo = qm(synthetic_batch(cfg))
    n_bins = len(cfg.model.decoder.speed_bins)
    print(f"query readout: {qm.num_parameters() / 1e6:.2f} M params, "
          f"waypoints {tuple(qo['waypoints'].shape)}, "
          f"speed logits {tuple(qo['speed_logits'].shape)}")
    if qo["waypoints"].shape != (2, T, 2) or qo["speed_logits"].shape != (2, n_bins):
        print("   FAIL: wrong output shapes"); ok = False
    if qm.decoder is not None:
        print("   FAIL: the GRU decoder is still allocated and never called")
        ok = False

    # The speed target must be the quantity the old controller computed, so that
    # switching readouts changes where the number comes from, not what it means.
    from egca.models.query_readout import speed_target
    wp = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]]])
    idx = speed_target(wp, cfg.model.decoder.wp_dt, qm.query_readout.speed_bins)
    want = float(qm.query_readout.speed_bins[idx[0]])
    implied = 3.0 / cfg.model.decoder.wp_dt          # 6 m/s from this trajectory
    print(f"speed target: trajectory implies {implied:.1f} m/s -> bin {want:.1f}")
    if abs(want - implied) > 1.0:
        print("   FAIL: speed target does not match the waypoint spacing")
        ok = False

    # ---- 2b. FiLM must be the identity at initialisation -------------------
    # The whole point of zero-initialising it: a conditioned run starts from the
    # same function as an unconditioned one, so a difference in the final score
    # is something the network learned rather than a different starting point.
    from egca.models.fusion import FiLM
    film = FiLM(256, 256)
    x = torch.randn(3, 17, 256)
    ego_vec = torch.randn(3, 256)
    if torch.allclose(film(x, ego_vec), x, atol=1e-6):
        print("FiLM is the identity at initialisation")
    else:
        print("   FAIL: FiLM perturbs the features before any training")
        ok = False

    # ---- 3. gradients must reach the new parameters ------------------------
    mdl = build(a.config, pooling="attention", gate_form="vector",
                ego_cond="film", camera_space="bev",
                goal_injection="both").train()
    o = mdl(synthetic_batch(cfg))
    o["waypoints"].sum().backward()
    # A zero-initialised FiLM still has to receive gradient on its weight: the
    # scale path is x * (1 + W.ego), so dL/dW is non-zero even though W is zero.
    dead = [n for n, p in mdl.named_parameters()
            if ("pool_" in n or "gate_mlp" in n or "film_" in n
                or "cam_to_bev" in n or "goal_embed" in n)
            and (p.grad is None or p.grad.abs().sum() == 0)]
    if dead:
        print(f"   FAIL: {len(dead)} new parameter tensors got no gradient")
        print("        e.g. " + ", ".join(dead[:3]))
        ok = False
    else:
        print("gradients reach every pooling, gate, FiLM and view-transform parameter")

    # ---- 4. every existing checkpoint must still load ----------------------
    print()
    ckpts = sorted(glob.glob(os.path.join(a.checkpoints, "*", "best.pth")))
    if not ckpts:
        print(f"no checkpoints under {a.checkpoints!r} -- skipping load check")
    for path in ckpts:
        name = os.path.basename(os.path.dirname(path))
        try:
            ck = torch.load(path, map_location="cpu")
            saved = ck.get("cfg")            # train.py stores it under "cfg"
            if saved is None:
                print(f"  {name:<12} no config stored -- cannot rebuild")
                ok = False
                continue
            from egca.models import EGCAPolicy
            from egca.config import Cfg
            m = EGCAPolicy(Cfg(saved), sensor_dropout=0.0)
            missing, unexpected = m.load_state_dict(ck["model"], strict=True), None
            print(f"  {name:<12} loads clean ({m.num_parameters() / 1e6:.1f} M)")
        except Exception as e:
            print(f"  {name:<12} FAILED: {type(e).__name__}: {str(e)[:70]}")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
