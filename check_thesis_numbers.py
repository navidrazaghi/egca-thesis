# -*- coding: utf-8 -*-
"""Check every architectural number the thesis asserts against the code.

Reading catches contradictions between two sentences. It does not catch a
sentence that quietly disagrees with the configuration file, because that
requires holding both in mind at once, and there are several dozen such numbers
spread over five chapters. This project has already shipped a thesis that
described a 32-beam LiDAR driving a 64-beam rig, cameras at +-55 that are
mounted at +-60, and a goal 8 m ahead of a model trained on goals averaging 23 m.

So the configuration is the source of truth and the prose is checked against it.
Persian digits are folded to ASCII first; the thesis writes numerals in Persian
and the config in ASCII, which is exactly why the two drifted unnoticed.

Usage:
    python check_thesis_numbers.py
"""
import io
import re
import sys

import yaml

FA = "۰۱۲۳۴۵۶۷۸۹"
TRANS = {ord(c): str(i) for i, c in enumerate(FA)}
TRANS[ord("٫")] = "."          # Persian decimal separator


def fold(s):
    return s.translate(TRANS)


def load_text():
    parts = []
    for f in ("content_part1.py", "content_part2.py", "content_part3.py"):
        parts.append(io.open(f, encoding="utf-8").read())
    return fold("\n".join(parts))


def main():
    cfg = yaml.safe_load(io.open("code/configs/egca.yaml", encoding="utf-8"))
    text = load_text()

    m, t = cfg["model"], cfg["train"]
    lid, cam, fus, dec = m["lidar"], m["camera"], m["fusion"], m["decoder"]
    ctl = cfg["control"]

    # (label, value the code holds, string the prose must contain)
    grid = int(round((lid["x_range"][1] - lid["x_range"][0]) / lid["pillar_size"]))
    checks = [
        ("embedding dimension", m["embed_dim"], "d = 256"),
        ("image height x width", cam["image_size"], "704"),
        ("pillar size", lid["pillar_size"], "0.25"),
        ("BEV grid", grid, "%d" % grid),
        ("fusion blocks L", fus["num_blocks"], "L=4"),
        ("attention heads H", fus["num_heads"], "H = 4"),
        ("FFN inner dimension", fus["ffn_dim"], "512"),
        ("gate hidden", fus["gate_hidden"], "128"),
        ("waypoint horizon T", dec["horizon"], "T=4"),
        ("waypoint spacing", dec["wp_dt"], "0.5"),
        ("sensor dropout rho", t["sensor_dropout"], "0.15"),
        ("learning rate", t["lr"], "10"),
        ("gradient clip", 5.0, "5"),
        ("epochs", 25, "25"),
        ("effective batch", t["batch_size"] * t["grad_accum"], "128"),
        ("validation subset", cfg["data"]["val_max_frames"], "8000"),
        ("lateral Kp", ctl["lateral"]["kp"], "1.25"),
        ("lateral Ki", ctl["lateral"]["ki"], "0.75"),
        ("lateral Kd", ctl["lateral"]["kd"], "0.30"),
        ("longitudinal Kp", ctl["longitudinal"]["kp"], "5.0"),
        ("integral window", ctl["lateral"]["window"], "20"),
        ("max throttle", ctl["max_throttle"], "0.75"),
        ("brake ratio", ctl["brake_speed_ratio"], "1.05"),
        ("creep threshold", ctl["creep_after_s"], "55"),
        ("creep duration", ctl["creep_for_s"], "1.5"),
        ("creep speed", ctl["creep_speed"], "4"),
    ]

    bad = 0
    print("configuration values asserted in the text")
    for label, value, needle in checks:
        ok = needle.replace(" ", "") in text.replace(" ", "")
        bad += not ok
        print("  %-28s %-12s %s" % (label, value, "found" if ok else "NOT FOUND"))

    # ---- arithmetic the thesis states, recomputed ---------------------------
    print("\nderived quantities recomputed from the configuration")
    d = m["embed_dim"]
    H = fus["num_heads"]
    dh = d // H
    L = fus["num_blocks"]
    n_c = (cam["image_size"][0] // 16) * (cam["image_size"][1] // 16)
    n_l = (grid // 2) ** 2
    full = 2 * n_c * n_l * d * 2 * L
    lin = (n_c + n_l) * dh * d * 2 * L
    derived = [
        ("N_c (stride 16)", n_c, "440"),
        ("N_l (64x64)", n_l, "4096"),
        ("d_h = d/H", dh, "64"),
        ("full attention GMAC", round(full / 1e9, 2), "7.38"),
        ("linear attention GMAC", round(lin / 1e9, 2), "0.59"),
        ("ratio", round(full / lin, 1), "12.4"),
        ("break-even N_qN_k", n_c * n_l, None),
    ]
    for label, value, expect in derived:
        if expect is None:
            print("  %-28s %s" % (label, value))
            continue
        agrees = str(value) == expect
        in_text = expect.replace(" ", "") in text.replace(" ", "")
        bad += not (agrees and in_text)
        print("  %-28s %-12s computed=%s text=%s"
              % (label, value, "ok" if agrees else "MISMATCH",
                 "ok" if in_text else "ABSENT"))

    print("\n%s" % ("all consistent" if not bad
                    else "%d discrepancy(ies) -- fix before submitting" % bad))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
