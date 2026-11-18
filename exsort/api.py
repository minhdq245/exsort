"""Configuration checks and tracker construction."""
import copy
import math
from .tracker import ExSort

PARAMS = ("DET", "INIT", "NMS", "IOU", "MIN_HITS", "MAX_AGE", "HEAD_FRACTION", "EV", "GLARE_COAST_SECONDS")
INTEGER = ("MIN_HITS", "MAX_AGE")


def validate(config):
    """Return a checked copy of {"params": {...}} with exactly the keys in PARAMS."""
    config = copy.deepcopy(config)
    if set(config) != {"params"}:
        raise ValueError('A configuration has one key, "params"')
    p = config["params"]
    if set(p) != set(PARAMS):
        raise ValueError(f"params must contain exactly: {', '.join(PARAMS)}")
    for k in PARAMS:
        if type(p[k]) not in (int, float) or not math.isfinite(p[k]):
            raise ValueError(f"Invalid number: {k}")
    for k in INTEGER:
        if int(p[k]) != p[k] or p[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
        p[k] = int(p[k])
    if not 0 <= p["DET"] <= p["INIT"] <= 1:
        raise ValueError("Require 0 <= DET <= INIT <= 1")
    for k in ("NMS", "IOU", "EV"):
        if not 0 <= p[k] <= 1:
            raise ValueError(f"{k} must be in [0, 1]")
    if not 0 < p["HEAD_FRACTION"] <= 1:
        raise ValueError("HEAD_FRACTION must be in (0, 1]")
    if p["GLARE_COAST_SECONDS"] < 0:
        raise ValueError("GLARE_COAST_SECONDS must be non-negative")
    return config


def require_fps(fps):
    if type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The glare output limit requires a positive sequence FPS")
    return fps


def create_tracker(config, *, fps=None):
    """One tracker per sequence. fps converts GLARE_COAST_SECONDS to frames."""
    return ExSort(require_fps(fps), **validate(config)["params"])
