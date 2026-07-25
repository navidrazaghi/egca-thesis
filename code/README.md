# EGCA: Efficient Gated Cross-Attention for Autonomous Driving

Reference implementation of the **EGCA** (Efficient Gated Cross-Attention) architecture
described in the Master's thesis:

> **Improving the Performance and Robustness of End-to-End Autonomous Driving Systems via
> Multimodal Learning and Optimal Fusion of Heterogeneous Sensor Information**  
> Navid Razaghi, MSc Thesis, Sharif University of Technology, 2026.

---

## Overview

EGCA is an end-to-end imitation learning policy for urban autonomous driving that:

- Fuses **RGB camera** (440 stride-16 tokens) and **LiDAR** (4096 BEV tokens) via
  **bidirectional linear cross-attention** — O(N_c + N_l) instead of O(N_c N_l) (Sec. 4-3-1)
- Uses a **sensor-reliability gate** to weight modalities (Eqs. 4.7-4.9, Sec. 4-3-2)
- Applies **modality-level sensor dropout** during training for robustness (Eq. 4.15)
- Predicts waypoints via a **GRU decoder** with multi-task auxiliary supervision (BEV segmentation, depth)
- Achieves **DS = 55.8** on CARLA Longest6, outperforming TransFuser/InterFuser/ReasonNet (Table 5-3)

---

## Installation

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. (Optional) Install CARLA 0.9.14 Python API for data collection / evaluation
pip install carla==0.9.14
```

**Note:** Training and inference work standalone (with synthetic data for testing); CARLA is only
needed for real data collection and closed-loop driving evaluation.

---

## Quick Start

### Smoke Test (Synthetic Data)

Verify installation and model architecture with random data (no CARLA required):

```bash
# Train for 2 epochs on 256 synthetic samples (batch is capped at 4 in this mode:
# the configured batch of 128 assumes 4 x 24 GB GPUs)
python -m egca.training.train --config configs/egca.yaml --synthetic 256

# Benchmark latency
python -m egca.eval.latency --config configs/egca.yaml
```

Expected output: `29.1 M parameters` (27.9 M at inference), `~39 ms/frame` on an
RTX 3090 (Table 5-6). The full-attention ablation of the same model runs at ~45 ms.

---

## Training

### 1. Data Collection (CARLA)

The privileged expert needs CARLA's own `agents` package (the global route
planner), which is **not** part of the `carla` pip wheel. Copy it out of the
simulator once:

```bash
docker cp carla-2000:/home/carla/PythonAPI/carla/agents ./agents
```

Launch the CARLA 0.9.14 server (headless is fine), then collect demonstrations:

```bash
# Collect 10 routes per town in diverse weather (Sec. 5-1)
for town in Town01 Town02 Town03 Town04 Town06; do
    python -m egca.carla_sim.collect_data --town $town --routes 10 \
        --output dataset/$town --traffic-density 0.2
done
```

This produces `dataset/{Town*}/route_*/rgb/*.jpg`, `lidar/*.npy`,
`measurements/*.json`, `bev_seg/*.png`, `depth/*.npy` and a `route.json`
per route.

### 1b. Build the waypoint labels

Labels are **not** written during collection. They are reconstructed from the
recorded ego poses, so that they are the trajectory the expert actually drove —
including braking for a red light and recovering from an injected steering
perturbation:

```bash
python -m egca.data.build_labels --root dataset
```

The dataset loader only uses frames that have a label, so this step is
mandatory before training.

### 2. Train EGCA

```bash
# Default config: 60 epochs, batch 128, 4 GPUs (Table 5-2)
python -m egca.training.train --config configs/egca.yaml

# Checkpoints saved to: checkpoints/egca/best.pth
```

**Ablations** (Table 5-5):

```bash
# Disable sensor dropout
python -m egca.training.train --config configs/egca.yaml --set train.sensor_dropout=0.0

# Disable sensor-reliability gate
python -m egca.training.train --config configs/egca.yaml --set model.fusion.gate=false

# Full attention (baseline)
python -m egca.training.train --config configs/egca.yaml \
    --set model.fusion.attention=full
```

---

## Evaluation

### Closed-Loop Driving (CARLA Longest6)

```bash
# Launch CARLA 0.9.14, then:
python -m egca.carla_sim.evaluate --config configs/egca.yaml \
    --checkpoint checkpoints/egca/best.pth --weather ClearNoon \
    --output results/egca_clear.json
```

Runs 36 routes (6 per town) and reports **DS / RC / IS** (Sec. 5-2).

**Weather robustness** (Table 5-4):

```bash
for w in ClearNoon WetNoon HardRainNoon FogMorning ClearNight HardRainNight; do
    python -m egca.carla_sim.evaluate ... --weather $w --output results/$w.json
done
```

**Sensor failure** (Table 5-5 ablation):

```bash
# permanent failure of one modality
python -m egca.carla_sim.evaluate ... --drop-sensor cam
python -m egca.carla_sim.evaluate ... --drop-sensor lidar

# intermittent LiDAR loss (Fig. 5-4)
python -m egca.carla_sim.evaluate ... --lidar-drop-rate 0.5
```

---

## Project Structure

```
code/
├── configs/egca.yaml           # full hyperparameters (Table 4-1, Table 5-2)
├── egca/
│   ├── models/
│   │   ├── camera_encoder.py  # ResNet-34 + FPN merge + 2-D pos. enc. (Sec. 4-2-1)
│   │   ├── lidar_encoder.py   # PointPillars (Sec. 4-2-2, Eq. 3.8-3.9)
│   │   ├── attention.py       # linear cross-attention (Eqs. 3.16-3.17, App. A)
│   │   ├── fusion.py          # EGCA blocks + gate + sensor dropout (Sec. 4-3)
│   │   ├── decoder.py         # GRU waypoint decoder (Sec. 4-4, Eqs. 4.10-4.12)
│   │   ├── heads.py           # auxiliary BEV seg / depth heads (Sec. 4-3-3)
│   │   └── model.py           # full EGCAPolicy (Fig. 4-1)
│   ├── training/
│   │   ├── train.py           # main training loop (Algorithm 1)
│   │   └── losses.py          # uncertainty-weighted multi-task loss (Eq. 4.16)
│   ├── data/dataset.py        # CARLA imitation dataset (Sec. 5-1)
│   ├── control/pid.py         # PID controllers (Sec. 4-5, Eqs. 4.13-4.14)
│   ├── eval/
│   │   ├── metrics.py         # DS / RC / IS (Sec. 5-2, Eqs. 5.1-5.4)
│   │   └── latency.py         # inference benchmark (Table 5-6)
│   └── carla_sim/
│       ├── collect_data.py    # expert data collection (Sec. 5-1)
│       ├── evaluate.py        # closed-loop Longest6 eval (Sec. 5-3)
│       ├── expert.py          # privileged rule-based planner
│       ├── sensors.py         # 3-cam stitching + 32-beam LiDAR (Appendix B)
│       └── weather.py         # 14 train + 2 OOD test conditions (Table 5-4)
└── requirements.txt
```

---

## Key Results (Thesis Chapter 5)

| **Metric**        | **EGCA** | TransFuser | InterFuser | ReasonNet |
|-------------------|----------|------------|------------|-----------|
| Driving Score ↑   | **55.8** | 47.1       | 49.4       | 52.7      |
| Route Completion ↑| **86.9** | 79.6       | 81.4       | 84.2      |
| Infraction Score ↑| **0.66** | 0.59       | 0.61       | 0.63      |

*(Table 5-3, CARLA Longest6 benchmark)*

**Weather robustness:** EGCA degrades by **15.9%** in out-of-distribution hard rain at
night, vs. **27.9%** for TransFuser (Table 5-4).

**Sensor robustness:** at a 50% LiDAR frame-drop EGCA keeps DS = 45.7, above the
camera-only floor of 37.7, while TransFuser collapses to 26.1 (Fig. 5-4).

**Efficiency:** versus the *same* architecture with full softmax attention, the linear
form is statistically indistinguishable in DS (55.8 vs 53.9, below the 3.2-point
significance threshold) while cutting fusion-module cost by 40% and end-to-end
latency by 13%.

---

## Citation

If you use this code, please cite:

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

---

## License

Research code accompanying the thesis. For academic and non-commercial use only.
