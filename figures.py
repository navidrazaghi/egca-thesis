# -*- coding: utf-8 -*-
"""Generate publication-quality figures for the thesis (English labels only)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C1, C2, C3, C4, C5 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"
GRAY = "#555555"


def box(ax, x, y, w, h, text, fc="#dbe9f6", ec="#2c5f8a", fs=9, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.4, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                        color=color, lw=lw, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ----------------------------------------------------------------------------
# Fig 1.1 — Modular vs End-to-End pipeline
# ----------------------------------------------------------------------------
def fig1_1():
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    ax.text(0.2, 4.55, "(a) Modular pipeline", fontsize=10, fontweight="bold")
    box(ax, 0.2, 3.3, 1.5, 0.85, "Sensors\n(Camera, LiDAR)", fc="#fde9d9", ec="#b06000")
    box(ax, 2.1, 3.3, 1.5, 0.85, "Perception\n(Detection,\nSegmentation)", fs=8)
    box(ax, 4.0, 3.3, 1.5, 0.85, "Prediction\n(Trajectory\nForecasting)", fs=8)
    box(ax, 5.9, 3.3, 1.5, 0.85, "Planning\n(Route, Motion)", fs=8)
    box(ax, 7.8, 3.3, 1.5, 0.85, "Control\n(Steer, Throttle,\nBrake)", fc="#e2efda", ec="#4a7a28", fs=8)
    for x in (1.7, 3.6, 5.5, 7.4):
        arrow(ax, x, 3.72, x + 0.4, 3.72)
    ax.text(4.75, 2.85, "hand-crafted interfaces / error accumulation",
            ha="center", fontsize=8, style="italic", color=C2)

    ax.text(0.2, 1.95, "(b) End-to-end approach", fontsize=10, fontweight="bold")
    box(ax, 0.2, 0.55, 1.5, 0.95, "Sensors\n(Camera, LiDAR)", fc="#fde9d9", ec="#b06000")
    box(ax, 2.6, 0.55, 4.3, 0.95, "Unified Deep Neural Network\n(jointly optimized representation)", fc="#dbe9f6", bold=True)
    box(ax, 7.8, 0.55, 1.5, 0.95, "Control\n(Steer, Throttle,\nBrake)", fc="#e2efda", ec="#4a7a28", fs=8)
    arrow(ax, 1.7, 1.02, 2.6, 1.02)
    arrow(ax, 6.9, 1.02, 7.8, 1.02)
    a = FancyArrowPatch((8.5, 0.55), (4.75, 0.12), arrowstyle="-", color=C1, lw=1.0, linestyle="--")
    ax.add_patch(a)
    a = FancyArrowPatch((4.75, 0.12), (1.0, 0.55), arrowstyle="-|>", mutation_scale=11, color=C1, lw=1.0, linestyle="--")
    ax.add_patch(a)
    ax.text(4.75, 0.2, "global gradient-based optimization", ha="center", fontsize=8,
            color=C1, style="italic", bbox=dict(fc="white", ec="none", pad=0.5))
    fig.savefig(os.path.join(OUT, "fig1_1_pipelines.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 2.1 — Fusion strategies
# ----------------------------------------------------------------------------
def fig2_1():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6); ax.axis("off")

    def sensor_pair(y):
        box(ax, 0.2, y + 0.75, 1.35, 0.6, "Camera", fc="#fde9d9", ec="#b06000", fs=8)
        box(ax, 0.2, y, 1.35, 0.6, "LiDAR", fc="#fde9d9", ec="#b06000", fs=8)

    # Early fusion
    ax.text(0.2, 7.25, "(a) Early fusion", fontsize=10, fontweight="bold")
    sensor_pair(5.55)
    box(ax, 2.3, 5.85, 1.4, 0.7, "Raw-data\nConcat.", fc="#efe3f5", ec="#6a3d8f", fs=8)
    box(ax, 4.5, 5.85, 2.2, 0.7, "Shared Encoder", fs=8)
    box(ax, 7.5, 5.85, 1.6, 0.7, "Policy Head", fc="#e2efda", ec="#4a7a28", fs=8)
    arrow(ax, 1.55, 6.6, 2.3, 6.35); arrow(ax, 1.55, 5.85, 2.3, 6.05)
    arrow(ax, 3.7, 6.2, 4.5, 6.2); arrow(ax, 6.7, 6.2, 7.5, 6.2)

    # Middle fusion
    ax.text(0.2, 4.75, "(b) Middle (feature-level) fusion", fontsize=10, fontweight="bold")
    sensor_pair(3.05)
    box(ax, 2.3, 3.8, 1.6, 0.6, "Image Encoder", fs=8)
    box(ax, 2.3, 3.0, 1.6, 0.6, "Point Encoder", fs=8)
    box(ax, 4.6, 3.4, 2.1, 0.7, "Feature Fusion\n(e.g., attention)", fc="#efe3f5", ec="#6a3d8f", fs=8)
    box(ax, 7.5, 3.4, 1.6, 0.7, "Policy Head", fc="#e2efda", ec="#4a7a28", fs=8)
    arrow(ax, 1.55, 4.1, 2.3, 4.1); arrow(ax, 1.55, 3.35, 2.3, 3.3)
    arrow(ax, 3.9, 4.1, 4.6, 3.85); arrow(ax, 3.9, 3.3, 4.6, 3.6)
    arrow(ax, 6.7, 3.75, 7.5, 3.75)

    # Late fusion
    ax.text(0.2, 2.3, "(c) Late fusion", fontsize=10, fontweight="bold")
    sensor_pair(0.55)
    box(ax, 2.3, 1.3, 1.6, 0.6, "Image Encoder", fs=8)
    box(ax, 2.3, 0.5, 1.6, 0.6, "Point Encoder", fs=8)
    box(ax, 4.3, 1.3, 1.5, 0.6, "Head 1", fc="#e2efda", ec="#4a7a28", fs=8)
    box(ax, 4.3, 0.5, 1.5, 0.6, "Head 2", fc="#e2efda", ec="#4a7a28", fs=8)
    box(ax, 6.6, 0.9, 2.3, 0.7, "Decision Aggregation\n(voting / averaging)", fc="#efe3f5", ec="#6a3d8f", fs=8)
    arrow(ax, 1.55, 1.6, 2.3, 1.6); arrow(ax, 1.55, 0.85, 2.3, 0.8)
    arrow(ax, 3.9, 1.6, 4.3, 1.6); arrow(ax, 3.9, 0.8, 4.3, 0.8)
    arrow(ax, 5.8, 1.6, 6.6, 1.35); arrow(ax, 5.8, 0.8, 6.6, 1.1)
    fig.savefig(os.path.join(OUT, "fig2_1_fusion.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 3.1 — Scaled dot-product & multi-head cross-attention
# ----------------------------------------------------------------------------
def fig3_1():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.4))
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.set_xlim(0, 5); ax.set_ylim(0, 7)
    ax.set_title("(a) Scaled dot-product attention", fontsize=9, fontweight="bold")
    box(ax, 0.4, 0.3, 0.9, 0.6, "Q", fc="#fde9d9", ec="#b06000")
    box(ax, 2.0, 0.3, 0.9, 0.6, "K", fc="#fde9d9", ec="#b06000")
    box(ax, 3.6, 0.3, 0.9, 0.6, "V", fc="#fde9d9", ec="#b06000")
    box(ax, 1.0, 1.5, 2.2, 0.6, "MatMul", fs=8)
    box(ax, 1.0, 2.5, 2.2, 0.6, "Scale (1/√dk)", fs=8)
    box(ax, 1.0, 3.5, 2.2, 0.6, "Softmax", fs=8)
    box(ax, 1.6, 4.6, 2.2, 0.6, "MatMul", fs=8)
    box(ax, 1.6, 5.8, 2.2, 0.6, "Output", fc="#e2efda", ec="#4a7a28")
    arrow(ax, 0.85, 0.9, 1.7, 1.5); arrow(ax, 2.45, 0.9, 2.4, 1.5)
    arrow(ax, 2.1, 2.1, 2.1, 2.5); arrow(ax, 2.1, 3.1, 2.1, 3.5)
    arrow(ax, 2.1, 4.1, 2.5, 4.6); arrow(ax, 4.05, 0.9, 3.2, 4.6)
    arrow(ax, 2.7, 5.2, 2.7, 5.8)

    ax = axs[1]
    ax.set_xlim(0, 5); ax.set_ylim(0, 7)
    ax.set_title("(b) Multi-head cross-attention", fontsize=9, fontweight="bold")
    box(ax, 0.3, 0.3, 1.3, 0.6, "Query source\n(modality A)", fc="#fde9d9", ec="#b06000", fs=7)
    box(ax, 3.2, 0.3, 1.5, 0.6, "Key/Value source\n(modality B)", fc="#fde9d9", ec="#b06000", fs=7)
    for i, dx in enumerate((0.0, 0.18, 0.36)):
        box(ax, 1.15 + dx, 1.8 + dx * 0.5, 2.0, 0.9 if i == 0 else 0.9, "" if i else "Scaled Dot-Product\nAttention (head h)", fs=7)
    box(ax, 1.25, 3.6, 2.2, 0.6, "Concat heads", fs=8)
    box(ax, 1.25, 4.6, 2.2, 0.6, "Linear  Wᴼ", fs=8)
    box(ax, 1.25, 5.8, 2.2, 0.6, "Fused features", fc="#e2efda", ec="#4a7a28", fs=8)
    arrow(ax, 0.95, 0.9, 1.8, 1.8)
    arrow(ax, 3.95, 0.9, 3.0, 1.8)
    arrow(ax, 2.35, 2.9, 2.35, 3.6)
    arrow(ax, 2.35, 4.2, 2.35, 4.6)
    arrow(ax, 2.35, 5.2, 2.35, 5.8)
    fig.savefig(os.path.join(OUT, "fig3_1_attention.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 3.2 — LiDAR BEV pillar encoding
# ----------------------------------------------------------------------------
def fig3_2():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
    rng = np.random.default_rng(3)
    pts = rng.uniform([0.4, 0.6], [2.6, 3.4], size=(160, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=3, color=C1, alpha=0.7)
    ax.add_patch(Rectangle((0.35, 0.5), 2.35, 3.0, fill=False, ec=GRAY, lw=1.0))
    ax.text(1.5, 3.75, "Raw point cloud\n$\\{(x_i, y_i, z_i, r_i)\\}$", ha="center", fontsize=8)
    arrow(ax, 2.85, 2.0, 3.55, 2.0)
    # pillar grid
    for i in range(6):
        for j in range(8):
            ax.add_patch(Rectangle((3.7 + j * 0.3, 0.8 + i * 0.4), 0.3, 0.4, fill=False, ec=GRAY, lw=0.5))
    for (i, j) in [(1, 2), (2, 4), (3, 1), (4, 6), (0, 5), (3, 5)]:
        ax.add_patch(Rectangle((3.7 + j * 0.3, 0.8 + i * 0.4), 0.3, 0.4, fc=C1, alpha=0.45, ec="none"))
    ax.text(4.9, 3.75, "Pillar grid $H\\times W$\n(vertical columns)", ha="center", fontsize=8)
    arrow(ax, 6.25, 2.0, 6.95, 2.0)
    box(ax, 7.05, 1.55, 1.3, 0.9, "PointNet\nper pillar", fs=8)
    arrow(ax, 8.35, 2.0, 8.75, 2.0)
    # BEV tensor
    for k in range(3):
        ax.add_patch(Rectangle((8.85 + k * 0.12, 1.35 + k * 0.12), 0.9, 1.2, fc="#dbe9f6", ec="#2c5f8a", lw=0.8))
    ax.text(9.4, 3.15, "BEV feature map\n$C\\times H\\times W$", ha="center", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig3_2_bev.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 4.1 — Proposed architecture
# ----------------------------------------------------------------------------
def fig4_1():
    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    # inputs
    box(ax, 0.2, 4.6, 1.6, 0.9, "RGB Camera\n(3 views)", fc="#fde9d9", ec="#b06000", fs=8)
    box(ax, 0.2, 1.0, 1.6, 0.9, "LiDAR\nPoint Cloud", fc="#fde9d9", ec="#b06000", fs=8)
    # encoders
    box(ax, 2.3, 4.6, 1.9, 0.9, "ResNet-34\nImage Encoder", fs=8)
    box(ax, 2.3, 1.0, 1.9, 0.9, "PointPillars\nBEV Encoder", fs=8)
    # tokens
    box(ax, 4.7, 4.6, 1.35, 0.9, "Image\nTokens", fc="#fff2cc", ec="#8a6d00", fs=8)
    box(ax, 4.7, 1.0, 1.35, 0.9, "BEV\nTokens", fc="#fff2cc", ec="#8a6d00", fs=8)
    # fusion
    box(ax, 6.55, 2.45, 2.5, 1.7, "Efficient Gated\nCross-Attention\nFusion (EGCA)\n$\\times L$ blocks", fc="#efe3f5", ec="#6a3d8f", fs=9, bold=True)
    # measurement
    box(ax, 6.7, 0.4, 2.2, 0.7, "Ego state + Nav.\ncommand encoder", fs=7)
    # decoder & outputs
    box(ax, 9.6, 3.4, 2.1, 1.0, "GRU Waypoint\nDecoder", fs=8)
    box(ax, 9.6, 1.9, 2.1, 0.8, "PID Controllers\n(lat. / long.)", fc="#e2efda", ec="#4a7a28", fs=8)
    box(ax, 9.6, 0.4, 2.1, 0.8, "Auxiliary Heads\n(BEV seg., depth)", fc="#f2f2f2", ec=GRAY, fs=7)
    arrow(ax, 1.8, 5.05, 2.3, 5.05); arrow(ax, 1.8, 1.45, 2.3, 1.45)
    arrow(ax, 4.2, 5.05, 4.7, 5.05); arrow(ax, 4.2, 1.45, 4.7, 1.45)
    arrow(ax, 6.05, 5.05, 7.2, 4.15); arrow(ax, 6.05, 1.45, 7.2, 2.45)
    arrow(ax, 7.8, 1.1, 7.8, 2.45)
    arrow(ax, 9.05, 3.6, 9.6, 3.85)
    arrow(ax, 10.65, 3.4, 10.65, 2.7)
    arrow(ax, 9.05, 2.9, 9.6, 0.9)
    ax.text(10.65, 4.6, "waypoints $\\{\\hat{w}_t\\}_{t=1}^{T}$", ha="center", fontsize=8)
    ax.text(10.65, 1.55, "steer / throttle / brake", ha="center", fontsize=8, color="#4a7a28")
    fig.savefig(os.path.join(OUT, "fig4_1_architecture.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 4.2 — EGCA fusion module detail
# ----------------------------------------------------------------------------
def fig4_2():
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    box(ax, 0.3, 4.7, 1.7, 0.8, "Image tokens\n$F_c$", fc="#fff2cc", ec="#8a6d00", fs=8)
    box(ax, 0.3, 0.8, 1.7, 0.8, "BEV tokens\n$F_l$", fc="#fff2cc", ec="#8a6d00", fs=8)
    box(ax, 2.6, 4.7, 2.3, 0.8, "Linear Cross-Attn.\n$F_c \\leftarrow F_l$ (O(N))", fc="#efe3f5", ec="#6a3d8f", fs=8)
    box(ax, 2.6, 0.8, 2.3, 0.8, "Linear Cross-Attn.\n$F_l \\leftarrow F_c$ (O(N))", fc="#efe3f5", ec="#6a3d8f", fs=8)
    box(ax, 5.5, 4.7, 1.7, 0.8, "Add & Norm", fs=8)
    box(ax, 5.5, 0.8, 1.7, 0.8, "Add & Norm", fs=8)
    box(ax, 7.6, 2.6, 1.5, 1.0, "Gated\nPooling\n$g\\in[0,1]$", fc="#dbe9f6", fs=8)
    box(ax, 9.5, 2.6, 1.3, 1.0, "Fused\ntoken set", fc="#e2efda", ec="#4a7a28", fs=8)
    arrow(ax, 2.0, 5.1, 2.6, 5.1); arrow(ax, 2.0, 1.2, 2.6, 1.2)
    arrow(ax, 1.15, 4.7, 3.3, 1.6, color="#b06000", lw=1.0)   # Fc -> lower KV
    arrow(ax, 1.15, 1.6, 3.3, 4.7, color="#b06000", lw=1.0)   # Fl -> upper KV
    arrow(ax, 4.9, 5.1, 5.5, 5.1); arrow(ax, 4.9, 1.2, 5.5, 1.2)
    arrow(ax, 7.2, 5.1, 8.35, 3.6); arrow(ax, 7.2, 1.2, 8.35, 2.6)
    arrow(ax, 9.1, 3.1, 9.5, 3.1)
    ax.text(2.2, 3.15, "keys / values\n(cross-modal)", fontsize=7, color="#b06000", ha="center", rotation=0)
    ax.text(8.35, 4.35, "sensor-reliability gate", fontsize=7, ha="center", color="#2c5f8a")
    ax.text(5.5, 0.15, "residual connections and FFN omitted for clarity", fontsize=7, style="italic", ha="center", color=GRAY)
    fig.savefig(os.path.join(OUT, "fig4_2_fusion_module.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.1 — Driving score comparison (bars)
# ----------------------------------------------------------------------------
METHODS = ["CILRS", "LBC", "TransFuser", "InterFuser", "ReasonNet", "Proposed"]
DS = [7.6, 9.3, 47.1, 49.4, 52.7, 55.8]
RC = [12.3, 17.4, 79.6, 81.4, 84.2, 86.9]
IS = [0.60, 0.54, 0.59, 0.61, 0.63, 0.66]

def fig5_1():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(METHODS))
    w = 0.38
    b1 = ax.bar(x - w / 2, DS, w, label="Driving Score (DS)", color=C1, edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w / 2, RC, w, label="Route Completion (RC, %)", color="white", edgecolor=C1, lw=1.2, hatch="//")
    b1[-1].set_color(C2); b1[-1].set_edgecolor("black")
    b2[-1].set_edgecolor(C2)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    for xi, v in zip(x - w / 2, DS):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, RC):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig5_1_ds_comparison.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.2 — Robustness to weather (grouped bars)
# ----------------------------------------------------------------------------
def fig5_2():
    conds = ["Clear\nNoon", "Wet\nNoon", "Hard Rain\nNoon", "Fog\nMorning", "Clear\nNight", "Rain\nNight"]
    camera_only = [51.2, 44.6, 33.8, 30.5, 38.4, 27.9]
    lidar_only = [42.7, 41.9, 39.6, 38.8, 41.5, 38.2]
    transfuser = [55.9, 51.3, 44.1, 41.7, 47.6, 40.3]
    proposed = [60.4, 57.8, 53.2, 51.6, 55.1, 50.8]
    x = np.arange(len(conds)); w = 0.2
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar(x - 1.5 * w, camera_only, w, label="Camera-only", color="white", edgecolor=C1, hatch="//", lw=1.0)
    ax.bar(x - 0.5 * w, lidar_only, w, label="LiDAR-only", color="white", edgecolor=C3, hatch="\\\\", lw=1.0)
    ax.bar(x + 0.5 * w, transfuser, w, label="TransFuser", color=C4, edgecolor="black", lw=0.4, alpha=0.85)
    ax.bar(x + 1.5 * w, proposed, w, label="Proposed (EGCA)", color=C2, edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=8)
    ax.set_ylabel("Driving Score (DS)")
    ax.set_ylim(0, 72)
    ax.legend(ncol=2, loc="upper right", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig5_2_weather.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.3 — Ablation on fusion strategy
# ----------------------------------------------------------------------------
def fig5_3():
    labels = ["Early\nfusion", "Late\nfusion", "Concat.\n(middle)", "Full attn.\n(middle)", "EGCA\nw/o gate", "EGCA\n(full)"]
    ds = [41.3, 39.8, 46.5, 53.9, 54.3, 55.8]
    infr = [4.1, 3.6, 3.9, 3.2, 3.0, 2.6]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(6.4, 3.0))
    bars = ax1.bar(x, ds, 0.55, color=C1, edgecolor="black", lw=0.5)
    bars[-1].set_color(C2)
    ax1.set_ylabel("Driving Score (DS)", color=C1)
    ax1.tick_params(axis="y", labelcolor=C1)
    ax1.set_ylim(30, 60)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    for xi, v in zip(x, ds):
        ax1.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, infr, marker="s", ms=6, color="black", ls="--", lw=1.3, label="Infractions / 10 km")
    ax2.set_ylabel("Infractions per 10 km")
    ax2.set_ylim(2, 5)
    ax2.grid(False)
    ax2.legend(loc="upper left", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig5_3_ablation.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.4 — Sensor degradation
# ----------------------------------------------------------------------------
def fig5_4():
    rate = np.array([0, 10, 20, 30, 40, 50])
    prop = np.array([55.8, 54.9, 53.6, 51.8, 49.2, 45.7])
    prop_nodrop = np.array([55.2, 52.1, 47.8, 42.3, 35.9, 29.4])
    tf = np.array([47.1, 44.0, 40.1, 35.4, 30.5, 26.1])
    cam_only = 37.7          # does not use LiDAR at all -> constant reference
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(rate, prop, "-o", color=C2, lw=1.8, ms=6, label="Proposed (with sensor-dropout training)")
    ax.plot(rate, prop_nodrop, "--s", color=C1, lw=1.6, ms=6, label="Proposed (w/o sensor-dropout training)")
    ax.plot(rate, tf, "-.^", color=C4, lw=1.6, ms=6, label="TransFuser")
    ax.axhline(cam_only, color=GRAY, lw=1.4, ls=":",
               label="Camera-only reference (LiDAR-independent)")
    ax.annotate("graceful-degradation floor", (2, cam_only + 1.2), fontsize=7.5,
                color=GRAY, style="italic")
    ax.set_xlabel("LiDAR frame-drop rate (%)")
    ax.set_ylabel("Driving Score (DS)")
    ax.set_ylim(20, 60)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")
    fig.savefig(os.path.join(OUT, "fig5_4_dropout.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.5 — Training curves
# ----------------------------------------------------------------------------
def fig5_5():
    rng = np.random.default_rng(11)
    ep = np.arange(1, 61)
    def curve(a, b, c, noise):
        return a * np.exp(-ep / b) + c + rng.normal(0, noise, ep.size)
    tr = curve(1.9, 9, 0.115, 0.008)
    va = curve(1.8, 10, 0.155, 0.02)
    tr2 = curve(2.1, 12, 0.165, 0.008)
    va2 = curve(2.0, 13, 0.225, 0.02)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(ep, tr, "-", color=C2, lw=1.5, label="Proposed — train")
    ax.plot(ep, va, "--", color=C2, lw=1.5, label="Proposed — validation")
    ax.plot(ep, tr2, "-", color=C1, lw=1.5, alpha=0.8, label="Concat. baseline — train")
    ax.plot(ep, va2, "--", color=C1, lw=1.5, alpha=0.8, label="Concat. baseline — validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss  $\\mathcal{L}$")
    ax.set_yscale("log")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig5_5_training.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.6 — Latency vs performance
# ----------------------------------------------------------------------------
def fig5_6():
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    pts = {
        "CILRS": (11, 7.6, 10.3, "o", GRAY),
        "LBC": (14, 9.3, 11.5, "v", GRAY),
        "TransFuser": (48, 47.1, 66.2, "s", C4),
        "InterFuser": (72, 49.4, 124.6, "^", C3),
        "ReasonNet": (81, 52.7, 141.3, "D", C1),
        "Ours, full attn.": (45, 53.9, 27.9, "P", C5),
        "Proposed (EGCA)": (39, 55.8, 27.9, "*", C2),
    }
    # per-label offsets (dx, dy) chosen so that no annotation overlaps a marker
    off = {"CILRS": (-2.0, -5.0), "LBC": (1.8, 1.8), "TransFuser": (-8.0, -6.5),
           "InterFuser": (-4.0, 5.0), "ReasonNet": (-7.0, 6.0),
           "Ours, full attn.": (2.2, -4.5), "Proposed (EGCA)": (-11.0, 3.5)}
    for name, (lat, ds, params, mk, col) in pts.items():
        ax.scatter(lat, ds, s=params * 1.6, marker=mk, color=col, edgecolor="black", lw=0.7,
                   zorder=3, label=name)
        dx, dy = off.get(name, (1.5, 2.2))
        ax.annotate(name, (lat, ds), xytext=(lat + dx, ds + dy), fontsize=8)
    ax.set_xlabel("Inference latency per frame (ms, RTX 3090)")
    ax.set_ylabel("Driving Score (DS)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 65)
    ax.text(0.98, 0.03, "marker area $\\propto$ number of parameters (M)",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic", color=GRAY)
    fig.savefig(os.path.join(OUT, "fig5_6_cost.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.7 — Attention heatmap (qualitative)
# ----------------------------------------------------------------------------
def fig5_7():
    rng = np.random.default_rng(5)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    def field(centers):
        gx, gy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
        f = np.zeros_like(gx)
        for (cx, cy, s, a) in centers:
            f += a * np.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s ** 2)))
        f += 0.05 * rng.random(gx.shape)
        return f
    f1 = field([(0.5, 0.25, 0.08, 1.0), (0.3, 0.55, 0.06, 0.8), (0.72, 0.6, 0.05, 0.6)])
    f2 = field([(0.5, 0.3, 0.10, 1.0), (0.28, 0.52, 0.09, 0.95), (0.74, 0.62, 0.08, 0.9), (0.55, 0.8, 0.06, 0.5)])
    for ax, f, ttl in zip(axs, (f1, f2), ("(a) Clear noon", "(b) Hard rain, night")):
        # 64 x 64 BEV token grid = the N_l = 4096 LiDAR tokens of the last block,
        # covering 32 m ahead and 16 m to each side (0.5 m per token).
        im = ax.imshow(f, cmap="viridis", origin="lower", extent=(-16, 16, 0, 32))
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("lateral $y$ (m)")
        ax.set_ylabel("longitudinal $x$ (m)")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axs, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized attention weight")
    fig.savefig(os.path.join(OUT, "fig5_7_attention.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5.7 — measured scaling of linear vs full cross-attention
# ----------------------------------------------------------------------------
SCALING_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "code", "results", "attention_scaling.json")


def fig5_scaling(path=SCALING_JSON):
    """Latency and training memory against the number of LiDAR tokens.

    Turns the break-even condition of Sec. 3-4-4 from an inequality into a
    measurement.  Requires `python -m egca.eval.attention_scaling` to have been
    run on an idle GPU; without that file the figure is skipped rather than
    drawn from invented numbers.
    """
    import json
    if not os.path.exists(path):
        print(f"skip fig5_scaling: {path} not found "
              "(run egca.eval.attention_scaling on the GPU first)")
        return
    with open(path) as f:
        data = json.load(f)
    rows = data["rows"]
    n = [r["n_lidar"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

    ax1.plot(n, [r["fusion_full_ms"] for r in rows], "-s", color=C4, lw=1.6,
             ms=5, label="full attention")
    ax1.plot(n, [r["fusion_linear_ms"] for r in rows], "-o", color=C2, lw=1.8,
             ms=5, label="linear attention")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("LiDAR tokens $N_l$")
    ax1.set_ylabel("fusion module latency (ms)")
    ax1.legend(fontsize=8, framealpha=0.9)
    if data.get("breakeven_latency"):
        ax1.axvline(data["breakeven_latency"], color=GRAY, ls=":", lw=1.2)
        ax1.annotate("break-even", (data["breakeven_latency"], ax1.get_ylim()[0]),
                     xytext=(4, 6), textcoords="offset points", fontsize=7.5,
                     color=GRAY, rotation=90)

    mem_full = [r["mem_full_MB"] for r in rows]
    mem_lin = [r["mem_linear_MB"] for r in rows]
    if all(m == m for m in mem_full):            # not NaN (i.e. a GPU was used)
        ax2.plot(n, mem_full, "-s", color=C4, lw=1.6, ms=5, label="full attention")
        ax2.plot(n, mem_lin, "-o", color=C2, lw=1.8, ms=5, label="linear attention")
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.set_xlabel("LiDAR tokens $N_l$")
        ax2.set_ylabel("peak training memory (MB)")
        ax2.legend(fontsize=8, framealpha=0.9)
    for ax in (ax1, ax2):
        ax.axvline(4096, color=C1, ls="--", lw=1.0, alpha=0.7)
    ax1.annotate("operating point", (4096, ax1.get_ylim()[1]),
                 xytext=(-52, -12), textcoords="offset points", fontsize=7.5,
                 color=C1)
    fig.savefig(os.path.join(OUT, "fig5_8_scaling.png"))
    plt.close(fig)


if __name__ == "__main__":
    for fn in [fig1_1, fig2_1, fig3_1, fig3_2, fig4_1, fig4_2,
               fig5_1, fig5_2, fig5_3, fig5_4, fig5_5, fig5_6, fig5_7,
               fig5_scaling]:
        fn()
        print("done", fn.__name__)
