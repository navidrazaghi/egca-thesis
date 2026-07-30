"""Exercise the creeping logic against the failure it was added to fix.

The trace from the first real evaluation showed the car leaning on an obstacle:
throttle pinned at 0.75, brake at zero, speed decaying to nothing, and the route
ending on the blocked timeout several minutes later.  Creeping is meant to break
exactly that, but it is stateful and easy to get subtly wrong -- firing while
the car is legally stopped at a red light would be worse than not firing at all.

Four situations, each with a definite right answer:
  1. driving normally            -> never creeps
  2. stopped, nothing ahead      -> creeps once the threshold passes
  3. stopped, obstacle ahead     -> never creeps, however long it waits
  4. obstacle clears             -> creeps only after it is gone

Usage:  python tools/check_creep.py --config configs/egca.yaml
"""
import argparse
import sys

import numpy as np


def run(ctrl, steps, speed_fn, hazard_fn, wp=(0.05, 0.0)):
    """Drive the controller for `steps` ticks; return the ticks it forced motion."""
    fired = []
    waypoints = np.array([[wp[0], wp[1]], [2 * wp[0], wp[1]],
                          [3 * wp[0], wp[1]], [4 * wp[0], wp[1]]])
    for t in range(steps):
        before = ctrl.creep_steps
        ctrl.step(waypoints, speed_fn(t), hazard=hazard_fn(t))
        if ctrl.creep_steps > before or (before > 0 and ctrl.creep_steps == 0):
            fired.append(t)
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egca.yaml")
    a = ap.parse_args()

    from egca.config import load_config
    from egca.control.pid import WaypointController

    cfg = load_config(a.config, [])
    hz = float(getattr(cfg.control, "control_hz", 20.0))
    after = int(float(getattr(cfg.control, "creep_after_s", 55.0)) * hz)
    horizon = after + int(4 * hz)
    ok = True

    def fresh():
        return WaypointController(cfg.control, cfg.model.decoder.wp_dt)

    # 1. moving normally -- creeping must never engage
    c = fresh()
    fired = run(c, horizon, lambda t: 5.0, lambda t: False, wp=(2.5, 0.0))
    print(f"1. driving at 5 m/s          -> creeps: {len(fired)}")
    if fired:
        print("   FAIL: interrupted a moving car"); ok = False

    # 2. stopped with a clear road -- must engage after the threshold
    c = fresh()
    fired = run(c, horizon, lambda t: 0.0, lambda t: False)
    print(f"2. stopped, road clear       -> creeps: {len(fired)}, "
          f"first at t={fired[0] if fired else None} (threshold {after})")
    if not fired:
        print("   FAIL: never recovered from the standstill"); ok = False
    elif fired[0] < after:
        # Firing on tick `after` is correct: the counter has been incremented on
        # every tick from 0, so by then the car has stood still for after + 1
        # ticks, a hair over the configured 55 s.
        print("   FAIL: fired before the threshold"); ok = False

    # 3. stopped with an obstacle -- must never engage
    c = fresh()
    fired = run(c, horizon, lambda t: 0.0, lambda t: True)
    print(f"3. stopped, obstacle ahead   -> creeps: {len(fired)}")
    if fired:
        print("   FAIL: nudged into an obstacle"); ok = False

    # 4. obstacle clears late -- must wait for it, then go
    c = fresh()
    clear_at = after + int(1 * hz)
    fired = run(c, horizon, lambda t: 0.0, lambda t: t < clear_at)
    print(f"4. obstacle clears at t={clear_at} -> creeps: {len(fired)}, "
          f"first at t={fired[0] if fired else None}")
    if not fired:
        print("   FAIL: never moved once the road cleared"); ok = False
    elif fired[0] < clear_at:
        print("   FAIL: moved while the obstacle was still there"); ok = False

    # what the controller actually commands while creeping
    c = fresh()
    for t in range(after + 2):
        steer, throttle, brake = c.step(
            np.array([[0.05, 0.0]] * 4), 0.0, hazard=False)
    print(f"\ncommand while creeping: throttle={throttle:.2f} brake={brake:.0f}")
    if throttle <= 0.0 or brake > 0.0:
        print("   FAIL: creeping does not actually command motion"); ok = False

    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
