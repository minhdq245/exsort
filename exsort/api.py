"""Configure the frozen research implementation (one configuration per process)."""
import copy
import math
from . import core

NUMERIC = {"DET", "LOW", "NMS", "IOU", "MIN_HITS", "MAX_AGE", "HEAD_FRACTION", "EV", "INIT"}
ENUMS = {
    "BIRTH_POLICY": {"axis", "mixed", "iom_active", "iou"},
    "COAST_MODE": {"legacy", "budget", "none", "uncertainty", "high_anchor", "last_high"},
    "MEASURE": {"raw", "posterior", "blend"}, "LOW_POLICY": {"legacy", "strict"},
    "ASSOCIATION": {"iou", "hmiou", "score"},
    "LOW_OUTPUT": {"all", "high_anchor", "prior"},
    "OUTPUT_FILTER": {"none", "tracktrack"}, "LOST_BIRTH": {"veto", "defer"},
}
BOOLEANS = {"ADMISSIBLE", "LIFECYCLE", "RECOVERY"}


def validate(config):
    config = copy.deepcopy(config)
    p = config["params"]
    if set(p) != NUMERIC | set(ENUMS) | BOOLEANS | {"NOISE"}:
        raise ValueError("Configuration keys must match configs/selected.json exactly")
    for k in NUMERIC:
        if not isinstance(p[k], (int, float)) or not math.isfinite(p[k]):
            raise ValueError(f"Invalid number: {k}")
    if not 0 <= p["LOW"] < p["DET"] <= 1:
        raise ValueError("Require 0 <= LOW < DET <= 1")
    for k in ("NMS", "IOU", "HEAD_FRACTION", "EV", "INIT"):
        if not 0 <= p[k] <= 1:
            raise ValueError(k)
    if p["HEAD_FRACTION"] == 0:
        raise ValueError("HEAD_FRACTION must be positive")
    for k in ("MIN_HITS", "MAX_AGE"):
        if int(p[k]) != p[k] or p[k] < 1:
            raise ValueError(k)
        p[k] = int(p[k])
    for k, choices in ENUMS.items():
        if p[k] not in choices:
            raise ValueError(k)
    for k in BOOLEANS:
        if not isinstance(p[k], bool):
            raise ValueError(k)
    if p["NOISE"] not in (False, True, "low_only"):
        raise ValueError("NOISE")
    fixed = config["fixed_structure"]
    expected = dict(explain=True, lamp=False, emit=True, free=True, birth=True,
                    inherit=False, own=False, reid=False, low=True, birth_lost=False,
                    diag_cap=False, diag_near=False, diag_close=False, gate="axis", rho=None)
    if fixed != expected:
        raise ValueError("This release supports the documented appearance-free fixed structure only")
    return config


def create_tracker(config):
    """Globals mirror the frozen snapshots; use processes for different configurations."""
    config = validate(config)
    for k, v in config["params"].items():
        setattr(core, k, v)
    core.OPTS.clear()
    core.OPTS.update(config["fixed_structure"])
    return core.EXSORT()
