"""Does the confidence gate actually shift weight when the camera degrades?

The thesis claims the gate falls from 0.63 in clear weather to 0.31 in night
rain. That number cannot be produced from this dataset: the published set
records no weather, neither in the measurements nor in the route names, so
splitting its frames by condition is not possible and the claim needs closed-loop
runs under chosen weather to test as stated.

What is testable offline is the mechanism underneath it, and paired on identical
frames rather than across conditions that differ in many ways at once: degrade
the camera input, hold everything else fixed, and see which way the gate moves.
A gate that means what the thesis says it means must fall. One that does not move
is not reading sensor quality, whatever its average happens to be.

Two degradations, both crude on purpose -- they stand in for a class of failure,
not for CARLA's renderer:
  darkness   scale luminance down, as at night
  occlusion  heavy additive noise, as with rain or a dirty lens

Usage:
    PYTHONPATH=. python tools/check_gate_response.py --ckpt checkpoints/egca_s0/best.pth
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="~/transfuser/data")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    from egca.config import Cfg
    from egca.data.transfuser_dataset import TransfuserDataset
    from egca.data.dataset import collate
    from egca.models import EGCAPolicy

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cfg = Cfg(ck["cfg"])
    model = EGCAPolicy(cfg, sensor_dropout=0.0).cuda().eval()
    model.load_state_dict(EGCAPolicy.upgrade_state_dict(ck["model"]), strict=True)
    print("checkpoint: %s (epoch %s)" % (a.ckpt, ck.get("epoch")))

    ds = TransfuserDataset(cfg, os.path.expanduser(a.root), [a.town],
                           augment=False, split="val")
    idx = np.linspace(0, len(ds) - 1, a.frames).astype(int)

    def degrade(img, kind, rng):
        if kind == "clean":
            return img
        if kind == "dark":
            return img * 0.18
        noise = torch.from_numpy(
            rng.normal(0.0, 40.0, size=tuple(img.shape)).astype(np.float32))
        return (img + noise.to(img.device)).clamp(0.0, 255.0)

    # "absent" is the one condition these runs were actually trained on: sensor
    # dropout removes a whole modality and substitutes the learned absent token.
    # Including it separates two very different claims -- that the gate tracks
    # sensor *quality*, which is what chapter 5 asserted, and that it tracks
    # sensor *presence*, which is what the training signal taught. Without this
    # row a flat response to degradation reads as "the gate does nothing", which
    # would be the wrong conclusion.
    kinds = ["clean", "dark", "noisy", "cam absent", "lidar absent"]
    gate = {k: [] for k in kinds}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            chunk = [ds[int(i)] for i in idx[s:s + a.batch]]
            base = collate(chunk)
            for k in kinds:
                b = {n: t.cuda() for n, t in base.items()}
                drop = None
                if k == "cam absent":
                    drop = "cam"
                elif k == "lidar absent":
                    drop = "lidar"
                else:
                    b["image"] = degrade(b["image"], k, rng)
                g = model(b, force_drop=drop)["gate"]
                if g is None:
                    print("this configuration has no gate; nothing to report")
                    return
                gate[k].extend(
                    np.atleast_1d(g.detach().float().cpu().numpy().reshape(-1)
                                  ).tolist())

    n = len(gate["clean"])
    print("paired over %d frames of %s\n" % (n, a.town))
    clean = np.array(gate["clean"])
    print("  %-9s %8s  %10s" % ("camera", "gate", "vs clean"))
    for k in kinds:
        v = np.array(gate[k])
        d = "" if k == "clean" else "%+.3f" % float((v - clean).mean())
        print("  %-9s %8.3f  %10s" % (k, float(v.mean()), d))

    print("\nThe gate weights the camera branch, so degrading the camera must "
          "push it\ndown. A flat response means the value is not tracking "
          "sensor quality,\nand the interpretability claim does not hold "
          "whatever its average is.")


if __name__ == "__main__":
    main()
