"""Configuration loading utilities."""
import copy
import yaml


class Cfg(dict):
    """Dict with attribute access, recursively."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def clone(self):
        return Cfg(copy.deepcopy(dict(self)))


def load_config(path, overrides=None):
    """Load a YAML config; `overrides` is a list of dotted key=value strings,
    e.g. ["model.fusion.attention=full", "train.sensor_dropout=0.0"]."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = Cfg(yaml.safe_load(f))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
