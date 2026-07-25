"""CARLA leaderboard metrics (Sec. 5-2), Eqs. (5.1)-(5.4).

Penalty coefficients follow the official CARLA leaderboard 1.0.
"""
from dataclasses import dataclass, field

PENALTIES = {
    "collision_pedestrian": 0.50,
    "collision_vehicle": 0.60,
    "collision_static": 0.65,
    "red_light": 0.70,
    "stop_sign": 0.80,
}


@dataclass
class RouteResult:
    completion: float                     # R_i in [0, 100]
    infractions: dict = field(default_factory=dict)   # name -> count
    distance_km: float = 0.0

    @property
    def infraction_score(self):
        """IS_i = prod_j p_j^{n_ij}  (Eq. 5.2)."""
        s = 1.0
        for name, count in self.infractions.items():
            s *= PENALTIES.get(name, 1.0) ** count
        return s

    @property
    def driving_score(self):
        return self.completion * self.infraction_score


def aggregate(results):
    """Benchmark-level RC / IS / DS (Eqs. 5.1-5.3) + infractions per 10 km."""
    n = len(results)
    rc = sum(r.completion for r in results) / n
    is_ = sum(r.infraction_score for r in results) / n
    ds = sum(r.driving_score for r in results) / n
    km = sum(r.distance_km for r in results)
    inf = sum(sum(r.infractions.values()) for r in results)
    return {"DS": ds, "RC": rc, "IS": is_,
            "infractions_per_10km": 10.0 * inf / max(km, 1e-9)}


def robustness_delta(ds_clear, ds_adverse):
    """Relative degradation, Eq. (5.4)."""
    return 100.0 * (ds_clear - ds_adverse) / max(ds_clear, 1e-9)
