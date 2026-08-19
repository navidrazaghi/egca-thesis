# EGCA — Efficient Gated Cross-Attention for End-to-End Driving

Master's thesis, Sharif University of Technology, 2026.

> **Improving the Performance and Robustness of End-to-End Autonomous Driving
> Systems via Multimodal Learning and Optimal Fusion of Heterogeneous Sensor
> Information**
> Navid Razaghi

This repository holds three things: the architecture and training code, the
thesis document and its build pipeline, and — in `docs/` — the measurements and
reasoning behind both.

---

## Status

Work in progress. Closed-loop Longest6, measured in this chain:

| run | routes | DS | RC | IS |
|---|---|---|---|---|
| privileged expert (reference agent) | 36 | 64.03 | 73.38 | 0.884 |
| `tf_base_rl` | 34 | 4.10 | 29.25 | 0.250 |
| `concat` ablation | 30 | 2.54 | 21.16 | 0.324 |

The published baselines are **not** re-run here; their figures are cited in the
thesis and no claim of superiority is made against them. What is compared is the
proposed method against the reference expert in the same chain, and ablations
against each other.

The policy drives — two routes completed outright, six above 50% — and fails by
colliding, 3.92 vehicle collisions per route. Three fixes for that are
implemented and committed but not yet trained or evaluated.

---

## Layout

```
code/            architecture, training, CARLA agent, verification tools
  egca/          model, data adapters, control, simulator interface
  tools/         one script per thing that had to be measured
  configs/       egca.yaml is the single source of hyperparameters
docs/            findings, lessons, evaluation protocol
figs/            figures used by the thesis
content_part*.py thesis text, built by build_thesis.py
```

Start with [`code/README.md`](code/README.md) to run anything.

---

## Documentation

**[docs/FINDINGS.md](docs/FINDINGS.md)** — every measured result and the tool
that produced it. The harness validation, four train/deploy convention
mismatches, a supervision target that is wrong on exactly the frames deciding
closed-loop behaviour, a reliability gate that responds to pooling statistics
rather than sensors, and two dead ends recorded so they are not retried.

**[docs/EVALUATION.md](docs/EVALUATION.md)** — how to run the benchmark, why
Longest6 needs the `*_local.py` evaluator, the screening protocol for cheap
iteration, and the traps that produce numbers which look valid and are not.

**[docs/LESSONS.md](docs/LESSONS.md)** — what generalises beyond this dataset.

The commit history is also written to be read: each message states what was
measured, what it showed, and what was decided.

---

## The short version of what was learned

A policy scoring 0.056 m open-loop scored DS 11.56 closed-loop, and the gap was
not the architecture. The published dataset's waypoint labels agree with the
pose the car actually reached to 0.080 m while it is moving, and disagree by a
factor of 5.6 on the frames where it pulls away from a standstill. A network fits
that faithfully and learns to creep when stopped.

Open-loop error cannot detect this, because it is scored against the same target
— it rewards the policy that reproduces the error more faithfully. Rebuilding the
target from realized poses raised the predicted pull-away speed from 0.34 to
1.64 m/s against the expert's 2.56, without disturbing the frames where the car
should stay stopped.

Our own collection never had the defect: its labels are derived from realized
poses and measure clean at 0.0000 m. The problem was in what we imported, and we
had audited our own pipeline without applying the same standard to the one we
adopted.

---

## Citation

```bibtex
@mastersthesis{razaghi2026egca,
  author = {Navid Razaghi},
  title  = {Improving the Performance and Robustness of End-to-End Autonomous
            Driving Systems via Multimodal Learning and Optimal Fusion of
            Heterogeneous Sensor Information},
  school = {Sharif University of Technology},
  year   = {2026},
  type   = {{MSc} Thesis}
}
```

Research code accompanying the thesis. Academic and non-commercial use.
