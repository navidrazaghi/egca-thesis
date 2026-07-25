"""Smoke test: verify model instantiation, forward pass, loss, and training
loop with synthetic data (no CARLA dependency).

Usage:
    cd code
    python smoke_test.py
"""
import sys
import torch

from egca.config import load_config
from egca.models import EGCAPolicy
from egca.data.dataset import CarlaDrivingDataset, collate
from egca.training.losses import UncertaintyWeightedLoss
from egca.control.pid import WaypointController
from egca.eval.metrics import RouteResult, aggregate


def test_model():
    print("1. Model instantiation...")
    cfg = load_config("configs/egca.yaml", [])
    model = EGCAPolicy(cfg, sensor_dropout=0.15)
    print(f"   parameters: {model.num_parameters() / 1e6:.1f} M")
    assert 20e6 < model.num_parameters() < 40e6, "param count out of range"


def test_forward():
    print("2. Forward pass (synthetic batch)...")
    cfg = load_config("configs/egca.yaml", [])
    model = EGCAPolicy(cfg, sensor_dropout=0.0).eval()
    ds = CarlaDrivingDataset(cfg, [], synthetic_len=8)
    batch = collate([ds[i] for i in range(4)])
    with torch.no_grad():
        out = model(batch)
    assert "waypoints" in out and out["waypoints"].shape == (4, 4, 2)
    assert "gate" in out and 0.0 <= out["gate"].mean() <= 1.0
    print(f"   waypoints shape: {out['waypoints'].shape}, gate mean: {out['gate'].mean():.3f}")


def test_loss():
    print("3. Loss computation...")
    cfg = load_config("configs/egca.yaml", [])
    model = EGCAPolicy(cfg, sensor_dropout=0.0).train()
    criterion = UncertaintyWeightedLoss(True, True)
    ds = CarlaDrivingDataset(cfg, [], synthetic_len=4)
    batch = collate([ds[i] for i in range(2)])
    out = model(batch)
    loss, parts = criterion(out, batch)
    assert "wp" in parts and "seg" in parts and "depth" in parts
    assert loss > 0
    print(f"   total loss: {loss:.4f}, parts: wp={parts['wp']:.3f} seg={parts['seg']:.3f} depth={parts['depth']:.3f}")


def test_training():
    print("4. Training loop (2 epochs, synthetic data)...")
    import subprocess
    ret = subprocess.call([sys.executable, "-m", "egca.training.train",
                           "--config", "configs/egca.yaml", "--synthetic", "64"],
                          cwd=".")
    assert ret == 0, "training script failed"


def test_control():
    print("5. PID controller...")
    cfg = load_config("configs/egca.yaml", [])
    ctrl = WaypointController(cfg.control, wp_dt=0.5)
    import numpy as np
    wps = np.array([[2.0, 0.1], [4.0, 0.2], [6.0, 0.3], [8.0, 0.4]])
    steer, throttle, brake = ctrl.step(wps, speed=3.0)
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    print(f"   steer={steer:.3f}, throttle={throttle:.3f}, brake={brake}")


def test_metrics():
    print("6. Metrics...")
    r1 = RouteResult(completion=85.0, infractions={"collision_vehicle": 1}, distance_km=1.2)
    r2 = RouteResult(completion=92.0, infractions={}, distance_km=1.5)
    agg = aggregate([r1, r2])
    assert 0 < agg["DS"] < 100
    print(f"   DS={agg['DS']:.1f}, RC={agg['RC']:.1f}, IS={agg['IS']:.2f}")


def test_latency():
    print("7. Latency benchmark...")
    import subprocess
    ret = subprocess.call([sys.executable, "-m", "egca.eval.latency",
                           "--config", "configs/egca.yaml"], cwd=".")
    assert ret == 0


if __name__ == "__main__":
    print("EGCA Smoke Test\n" + "="*50)
    test_model()
    test_forward()
    test_loss()
    test_control()
    test_metrics()
    test_training()
    test_latency()
    print("="*50 + "\nAll tests passed OK")
