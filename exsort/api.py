"""Configure the frozen research implementation (one configuration per process)."""
import copy
import math
from . import core

NUMERIC = {"DET", "LOW", "NMS", "IOU", "MIN_HITS", "MAX_AGE", "HEAD_FRACTION", "EV", "INIT"}
COAST_LIMITS = {"BLIND_COAST_SECONDS", "GLARE_COAST_SECONDS"}
ENUMS = {
    "BIRTH_POLICY": {"axis", "mixed", "iom_active", "iou"},
    "COAST_MODE": {"legacy", "budget", "fps_budget", "none", "uncertainty", "high_anchor", "last_high"},
    "MEASURE": {"raw", "posterior", "blend"}, "LOW_POLICY": {"legacy", "strict"},
    "ASSOCIATION": {"iou", "hmiou", "score"},
    "LOW_OUTPUT": {"all", "high_anchor", "prior"},
    "OUTPUT_FILTER": {"none", "tracktrack"}, "LOST_BIRTH": {"veto", "defer"},
}
BOOLEANS = {"ADMISSIBLE", "LIFECYCLE", "RECOVERY"}
# Optional keys; when absent the original behaviour is used (see core.CREDIBILITY).
OPTIONAL = {"CREDIBILITY": False}


def validate(config):
    config = copy.deepcopy(config)
    p = config["params"]
    required = NUMERIC | set(ENUMS) | BOOLEANS | {"NOISE"}
    if set(p) - set(OPTIONAL) not in (required, required | COAST_LIMITS):
        raise ValueError("Use the documented configuration keys and specify both coast limits together")
    for k in NUMERIC:
        if not isinstance(p[k], (int, float)) or not math.isfinite(p[k]):
            raise ValueError(f"Invalid number: {k}")
    if COAST_LIMITS <= p.keys():
        for k in COAST_LIMITS:
            if type(p[k]) not in (int, float) or not math.isfinite(p[k]) or p[k] < 0:
                raise ValueError(f"{k} must be a finite non-negative duration")
        if p["COAST_MODE"] != "fps_budget":
            raise ValueError("Coast limits in seconds require fps_budget mode")
        if p["GLARE_COAST_SECONDS"] < p["BLIND_COAST_SECONDS"]:
            raise ValueError("The glare-assisted coast limit must be at least the blind limit")
    if not 0 <= p["LOW"] <= p["DET"] <= 1:
        raise ValueError("Require 0 <= LOW <= DET <= 1 (LOW == DET matches every kept detection in one pass)")
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
    if not isinstance(p.get("CREDIBILITY", False), bool):
        raise ValueError("CREDIBILITY")
    if p["NOISE"] not in (False, True, "low_only"):
        raise ValueError("NOISE")
    fixed = config["fixed_structure"]
    expected = dict(explain=True, lamp=False, emit=True, free=True, birth=True,
                    inherit=False, own=False, reid=False, low=True, birth_lost=False,
                    diag_cap=False, diag_near=False, diag_close=False, gate="axis", rho=None)
    if fixed != expected:
        raise ValueError("This release supports the documented appearance-free fixed structure only")
    return config


def require_fps(fps):
    if type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The FPS-based output policy requires a positive sequence FPS")
    return fps


def create_tracker(config, *, fps=None):
    """Globals mirror the frozen snapshots; use processes for different configurations."""
    config = validate(config)
    params = config["params"]
    if params["COAST_MODE"] == "fps_budget":
        fps = require_fps(fps)
    blind_seconds = params.pop("BLIND_COAST_SECONDS", 1.0)
    glare_seconds = params.pop("GLARE_COAST_SECONDS", 2.0)
    for k, v in {**OPTIONAL, **params}.items():
        setattr(core, k, v)
    core.OPTS.clear()
    core.OPTS.update(config["fixed_structure"])
    return core.EXSORT(fps=fps, blind_coast_seconds=blind_seconds, glare_coast_seconds=glare_seconds)
