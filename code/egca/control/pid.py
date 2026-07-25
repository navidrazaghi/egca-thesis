"""Low-level PID controllers (Sec. 4-5), Eqs. (4.13)-(4.14).
Gains follow Table 4-1 (TransFuser reference values).
"""
from collections import deque

import numpy as np


class PID:
    """Discrete PID with a *windowed* integral term (Eq. 4.13).

    The integral is the running mean of the last `window` errors rather than an
    unbounded sum: over long routes an unbounded accumulator saturates the
    actuator (integral wind-up), and the mean form keeps the three gains
    dimensionless and independent of the control period.
    """

    def __init__(self, kp, ki, kd, window=20):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.err = deque(maxlen=window)

    def reset(self):
        self.err.clear()

    def step(self, e):
        self.err.append(e)
        integral = sum(self.err) / len(self.err)
        deriv = self.err[-1] - self.err[-2] if len(self.err) > 1 else 0.0
        return self.kp * e + self.ki * integral + self.kd * deriv


class WaypointController:
    """Converts predicted ego-frame waypoints to (steer, throttle, brake)."""

    def __init__(self, ctrl_cfg, wp_dt=0.5):
        lat, lon = ctrl_cfg.lateral, ctrl_cfg.longitudinal
        self.turn = PID(lat.kp, lat.ki, lat.kd, lat.window)
        self.speed = PID(lon.kp, lon.ki, lon.kd, lon.window)
        self.max_throttle = ctrl_cfg.max_throttle
        self.brake_ratio = ctrl_cfg.brake_speed_ratio
        self.clip_delta = ctrl_cfg.clip_delta
        self.wp_dt = wp_dt

    def reset(self):
        self.turn.reset()
        self.speed.reset()

    def step(self, waypoints, speed):
        """waypoints: T x 2 (x fwd, y left) in ego frame; speed in m/s."""
        wp = np.asarray(waypoints, dtype=np.float64)
        # -- lateral: heading error toward the aim point (Eq. 4.13).  The error
        # is negated because a target on the left (y > 0) requires a negative
        # CARLA steering command; it is normalized by 90 deg so that the gains
        # of Table 4-1 are dimensionless.
        aim = (wp[0] + wp[1]) / 2.0
        angle = -np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        steer = float(np.clip(self.turn.step(angle), -1.0, 1.0))
        # -- longitudinal: desired speed from waypoint spacing (Eq. 4.14)
        v_des = float(np.linalg.norm(wp[1] - wp[0]) / self.wp_dt)
        brake = speed > v_des * self.brake_ratio or v_des < 0.4
        delta = np.clip(v_des - speed, 0.0, self.clip_delta)
        throttle = float(np.clip(self.speed.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0
        return steer, throttle, float(brake)
