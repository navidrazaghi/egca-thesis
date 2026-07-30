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
        # Creeping (TransFuser Sec. 3.5).  Standing still is overwhelmingly the
        # most common thing to do next in a demonstration set collected in dense
        # traffic, so a cloned policy that stops can fail to ever start again --
        # the inertia problem.  Our own traces show the same shape: throttle
        # pinned at 0.75, brake at zero, speed decaying to nothing while the car
        # leans on an obstacle until the blocked timeout ends the route.
        hz = float(getattr(ctrl_cfg, "control_hz", 20.0))
        self.creep_after = int(float(getattr(ctrl_cfg, "creep_after_s", 55.0)) * hz)
        self.creep_for = int(float(getattr(ctrl_cfg, "creep_for_s", 1.5)) * hz)
        self.creep_speed = float(getattr(ctrl_cfg, "creep_speed", 4.0))
        self.stuck_steps = 0
        self.creep_steps = 0

    def reset(self):
        self.turn.reset()
        self.speed.reset()
        self.stuck_steps = 0
        self.creep_steps = 0

    def _creeping(self, speed, hazard):
        """Decide whether to force the car forward this step.

        The hazard flag comes from the caller because it is a sensor question,
        not a control one.  Creeping without it is unsafe: the reason the car is
        stationary is often that something is directly in front of it.
        """
        if speed < 0.1:
            self.stuck_steps += 1
        elif self.creep_steps == 0:
            self.stuck_steps = 0
        if self.stuck_steps <= self.creep_after:
            return False
        if hazard:                      # blocked for a real reason; keep waiting
            self.creep_steps = 0
            return False
        self.creep_steps += 1
        if self.creep_steps >= self.creep_for:
            self.stuck_steps = 0        # one nudge, then re-arm
            self.creep_steps = 0
        return True

    def step(self, waypoints, speed, hazard=False, v_des=None):
        """waypoints: T x 2 (x fwd, y left) in ego frame; speed in m/s.

        `v_des` overrides the speed implied by the waypoint spacing.  With the
        query readout the network predicts it from the scene, which breaks the
        loop that made a standstill self-sustaining: previously a collapsed
        trajectory meant zero desired speed, zero speed meant a stationary input,
        and a stationary input reproduced the collapsed trajectory.
        """
        wp = np.asarray(waypoints, dtype=np.float64)
        # -- lateral: heading error toward the aim point (Eq. 4.13).  The error
        # is negated because a target on the left (y > 0) requires a negative
        # CARLA steering command; it is normalized by 90 deg so that the gains
        # of Table 4-1 are dimensionless.
        aim = (wp[0] + wp[1]) / 2.0
        angle = -np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        steer = float(np.clip(self.turn.step(angle), -1.0, 1.0))
        # -- longitudinal: desired speed from waypoint spacing (Eq. 4.14),
        # unless the network predicted it directly
        if v_des is None:
            v_des = float(np.linalg.norm(wp[1] - wp[0]) / self.wp_dt)
        else:
            v_des = float(v_des)
        if self._creeping(speed, hazard):
            # Override the network: it is predicting a standstill and has no way
            # out of it, because a standstill is what it was shown.
            v_des = self.creep_speed
            brake = False
        else:
            brake = speed > v_des * self.brake_ratio or v_des < 0.4
        delta = np.clip(v_des - speed, 0.0, self.clip_delta)
        throttle = float(np.clip(self.speed.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0
        return steer, throttle, float(brake)
