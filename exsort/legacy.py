"""EX-SORT: SORT/OC-SORT with explained absence. One principle replaces HALO's rule set:
a track's missing detection is either *explained* by measured glare (a saturated headlamp in the head band of its
predicted box) or not. Explained absences do not age the track, move its centre to the lamp, and may be emitted;
unexplained absences age it exactly like SORT. Every identity decision (birth suppression, identity inheritance,
coast ownership) reuses the one association gate IOU, applied either to the boxes or to the boxes at equal size.

Constants: the five SORT/OC-SORT settings (DET, NMS, IOU, MIN_HITS, MAX_AGE), HEAD_FRACTION (where a helmet lamp sits
in a person box) and EV = .5 (the MOT/CLEAR match threshold, used as the coast-emission test).
"""
import numpy as np
from .vendor.ocsort.association import iou_batch, linear_assignment
from .vendor.ocsort.ocsort import KalmanBoxTracker
from .geometry import norm_box_to_xyxy, xyxy_to_norm_box

DET = 0.25
NMS = 0.60
IOU = 0.30
MIN_HITS = 3
MAX_AGE = 60
HEAD_FRACTION = 0.25
EV = 0.50
LOW = 0.10
AW = 0.50       # appearance weight in the association cost (TrackTrack value), only with OPTS["reid"]
TAU = -1.0      # birth veto: a geometric claim only suppresses a birth if cosine(det, track) >= TAU (-1 = off)
ALPHA = 0.95    # appearance EMA (TrackTrack value)
MASK_S = 512

# ablation switches (each removes a mechanism; none adds a constant)
OPTS = {"explain": True, "lamp": True, "emit": True, "free": True, "birth": True, "inherit": True, "own": True,
        "reid": False, "low": False, "birth_lost": False, "diag_cap": False, "diag_near": False, "diag_close": False, "gate": "axis", "rho": None}
# diagnostic only (to locate gaps against halo3a; not part of the method)
DIAG = {"GLARE_MAX_AGE": 160, "DUP": 0.35, "NEAR_IOU": 0.20, "NEAR": 0.55}


def config():
    return {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float)) and k != "MASK_S"}


def _nms_keep(D):
    if len(D) <= 1:
        return list(range(len(D)))
    keep = []
    for i in np.argsort(-D[:, 4]):
        if all(iou_batch(D[i:i + 1, :4], D[j:j + 1, :4])[0, 0] <= NMS for j in keep):
            keep.append(int(i))
    return sorted(keep)


def _nms(D):
    return D[_nms_keep(D)]


def _assoc(iou, Fd, Ft):
    """Hungarian on IoU, or on (1-AW)*(1-IoU) + AW*cosine distance when appearance features exist; IoU >= IOU stays a hard gate."""
    if Fd is None or Ft is None:
        return _hungarian(iou)
    if iou.size == 0 or iou.max() < IOU:
        return []
    cost = (1 - AW) * (1 - iou) + AW * np.clip(1 - Fd @ Ft.T, 0., 1.)
    return [(int(d), int(t)) for d, t in linear_assignment(cost) if iou[d, t] >= IOU]


def _hungarian(iou):
    if iou.size == 0 or iou.max() < IOU:
        return []
    return [(int(d), int(t)) for d, t in linear_assignment(-iou) if iou[d, t] >= IOU]


def _shape_free_iou(b, ref):
    """IoU of box b with itself moved to ref's centre: the overlap two boxes would have at equal size."""
    w, h = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
    dx = abs(0.5 * (b[0] + b[2] - ref[0] - ref[2])); dy = abs(0.5 * (b[1] + b[3] - ref[1] - ref[3]))
    i = max(w - dx, 0.0) * max(h - dy, 0.0)
    return i / (2 * w * h - i)


def _axis_claim(b, ref):
    """Centre distance below rho * (longer side), rho = (1 - IOU) / (1 + IOU): the shift along a person's long axis at
    which two equal boxes fall out of the association gate IOU."""
    rho = OPTS["rho"] if OPTS["rho"] is not None else (1.0 - IOU) / (1.0 + IOU)
    c = 0.5 * (b[:2] + b[2:4]) - 0.5 * (ref[:2] + ref[2:4])
    return float(np.hypot(*c)) < rho * max(b[2] - b[0], b[3] - b[1], ref[2] - ref[0], ref[3] - ref[1], 1.0)


def _lamp(sat, b, W, H):
    """Centroid of saturated pixels in the head band of box b, or None."""
    x1, y1, x2, y2 = b
    h = y2 - y1
    if x2 - x1 <= 1 or h <= 1:
        return None
    sx, sy = MASK_S / W, MASK_S / H
    a, c = int(max(0, x1 * sx)), int(min(MASK_S, np.ceil(x2 * sx)))
    d, e = int(max(0, y1 * sy)), int(min(MASK_S, np.ceil((y1 + HEAD_FRACTION * h) * sy)))
    if c <= a or e <= d:
        return None
    ys, xs = np.nonzero(sat[d:e, a:c])
    if xs.size == 0:
        return None
    return (xs.mean() + a + 0.5) / sx, (ys.mean() + d + 0.5) / sy


def _set_centre(kf, zx, zy):
    """The lamp fixes the box centre: Kalman update on the centre with a noiseless measurement (Joseph form)."""
    Hm = kf.H[:2, :]
    y = np.array([[zx], [zy]], np.float64) - Hm @ kf.x
    PHT = kf.P @ Hm.T
    K = PHT @ np.linalg.inv(Hm @ PHT)
    kf.x = kf.x + K @ y
    I_KH = np.eye(kf.dim_x) - K @ Hm
    kf.P = I_KH @ kf.P @ I_KH.T


class Track:
    def __init__(self, tid, row, W, H):
        self.id = tid
        self.kf = KalmanBoxTracker(row[:4])
        self.kf.update(np.asarray(row[:5], np.float32).copy())
        self.W, self.H = W, H
        self.box = row[:4].copy()
        self.unexplained = 0      # consecutive-loss frames not explained by glare
        self.now = False          # absence explained in the current frame
        self.seen = False         # confirmed identity (has been emitted)
        self.latched = False      # diagnostic: explained at some point during the current loss
        self.feature = None       # appearance EMA (OPTS["reid"])

    def pred(self):
        return self.kf.get_state()[0][:4].astype(np.float32)

    def out_box(self):
        b = self.box if self.kf.time_since_update == 0 else self.kf.get_state()[0]
        return xyxy_to_norm_box(b, self.W, self.H)

    def observe(self, row):
        self.kf.update(np.asarray(row[:5], np.float32).copy())
        self.box = row[:4].copy()
        self.unexplained, self.now, self.latched = 0, False, False

    def alive(self):
        if OPTS["diag_cap"]:
            return self.kf.time_since_update <= (DIAG["GLARE_MAX_AGE"] if self.latched else MAX_AGE)
        lost = self.unexplained if OPTS["free"] else self.kf.time_since_update
        return lost <= MAX_AGE


class EXSORT:
    def __init__(self):
        self.tracks, self.n, self.frame = [], 0, 0

    def _claims(self, b, t):
        if OPTS["diag_near"]:
            for r in (t.pred(), t.box):
                v = float(iou_batch(np.asarray(b, np.float32)[None], np.asarray(r, np.float32)[None])[0, 0])
                if v >= DIAG["DUP"]:
                    return True
                if v >= DIAG["NEAR_IOU"]:
                    c = 0.5 * (b[:2] + b[2:4]) - 0.5 * (r[:2] + r[2:4])
                    sz = max(b[2] - b[0], b[3] - b[1], r[2] - r[0], r[3] - r[1], 1.0)
                    if float(np.hypot(*c)) < DIAG["NEAR"] * sz:
                        return True
            return False
        if OPTS["gate"] == "shapefree":
            return any(_shape_free_iou(b, r) >= IOU for r in (t.pred(), t.box))
        return any(_axis_claim(b, r) for r in (t.pred(), t.box))

    def step(self, dets, W, H, frame_idx, sat=None):
        self.frame += 1
        rows = [list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']] for d in dets if d['conf'] >= DET]
        acc = [d for d in dets if d['conf'] >= DET]
        D = np.asarray(rows, np.float32).reshape(-1, 5)
        keep = _nms_keep(D); D = D[keep]
        F = np.asarray([acc[i]['feature'] for i in keep], np.float32).reshape(len(keep), -1) if OPTS["reid"] and keep else None
        ftr = lambda ts: np.asarray([self.tracks[t].feature for t in ts], np.float32) if F is not None and all(self.tracks[t].feature is not None for t in ts) else None
        lows = np.asarray([list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']] for d in dets if LOW <= d['conf'] < DET], np.float32).reshape(-1, 5) if OPTS["low"] else None
        P = np.asarray([t.kf.predict()[0][:4] for t in self.tracks], np.float32).reshape(-1, 4)
        ud, ut = list(range(len(D))), list(range(len(P)))
        matches = []
        if ud and ut:                                   # stage 1: predictions
            m = _assoc(iou_batch(D[:, :4], P), F, ftr(ut)); matches += m
            ud = sorted(set(ud) - {d for d, _ in m}); ut = sorted(set(ut) - {t for _, t in m})
        if ud and ut:                                   # stage 2: last observations (OC-SORT OCR)
            L = np.asarray([self.tracks[t].box for t in ut], np.float32)
            m = _assoc(iou_batch(D[ud, :4], L), None if F is None else F[ud], ftr(ut)); matches += [(ud[d], ut[t]) for d, t in m]
            ud = sorted(set(ud) - {ud[d] for d, _ in m}); ut = sorted(set(ut) - {ut[t] for _, t in m})
        for d, t in matches:
            self.tracks[t].observe(D[d])
            if F is not None:
                tr = self.tracks[t]
                if tr.feature is None:
                    tr.feature = F[d].copy()
                else:
                    beta = ALPHA + (1 - ALPHA) * (1 - D[d, 4]); f = beta * tr.feature + (1 - beta) * F[d]; tr.feature = f / np.linalg.norm(f)
        if lows is not None and len(lows) and ut:       # stage 3 (ByteTrack): low-confidence boxes may only continue unmatched tracks
            m = _hungarian(iou_batch(lows[:, :4], np.asarray([self.tracks[t].pred() for t in ut], np.float32)))
            for d, k in m:
                self.tracks[ut[k]].observe(lows[d])
            ut = [t for i, t in enumerate(ut) if i not in {k for _, k in m}]

        for t in ut:                                    # absences: explained or not
            tr = self.tracks[t]
            tr.kf.update(None)
            tr.now = False
            if OPTS["explain"] and sat is not None and tr.kf.hits >= MIN_HITS:
                pr = tr.pred()
                p = _lamp(sat, pr, W, H)
                if p is not None:
                    tr.now = tr.latched = True
                    if OPTS["lamp"]:
                        _set_centre(tr.kf.kf, p[0], p[1] + 0.5 * (1.0 - HEAD_FRACTION) * (pr[3] - pr[1]))
            if not tr.now:
                tr.unexplained += 1

        for d in ud:                                    # births: only detections no track can claim
            r = D[d]
            if OPTS["birth"] and any(self._claims(r[:4], t) and (F is None or TAU <= -1 or t.feature is None or float(F[d] @ t.feature) >= TAU)
                                     for t in self.tracks if not OPTS["birth_lost"] or t.kf.time_since_update > 0):
                continue
            self.n += 1
            self.tracks.append(Track(self.n, r, W, H))
            if F is not None:
                self.tracks[-1].feature = F[d].copy()
        if OPTS["inherit"]:
            self._inherit()
        self.tracks = [t for t in self.tracks if t.alive()]
        out = [t for t in self.tracks if (t.kf.hits >= MIN_HITS or self.frame <= MIN_HITS)
               and (t.kf.time_since_update == 0 or (OPTS["emit"] and t.now and self._localised(t)))]
        if OPTS["own"]:
            obs = np.asarray([t.box for t in out if t.kf.time_since_update == 0], np.float32).reshape(-1, 4)
            out = [t for t in out if t.kf.time_since_update == 0 or not len(obs)
                   or iou_batch(t.pred()[None], obs).max() < IOU]
        for t in self.tracks:
            if t.kf.hits >= MIN_HITS:
                t.seen = True
        return out

    @staticmethod
    def _localised(t):
        return iou_batch(t.pred()[None], np.asarray(t.box, np.float32)[None])[0, 0] >= EV

    @staticmethod
    def _close(b, t):
        for r in (t.pred(), t.box):
            c = 0.5 * (b[:2] + b[2:4]) - 0.5 * (r[:2] + r[2:4])
            if float(np.hypot(*c)) < DIAG["NEAR"] * max(b[2] - b[0], b[3] - b[1], r[2] - r[0], r[3] - r[1], 1.0):
                return True
        return False

    def _inherit(self):
        """A track confirmed this frame that exactly one lost confirmed track can claim continues that identity."""
        if self.frame <= MIN_HITS:
            return
        gone = set()
        for nt in [t for t in self.tracks if t.kf.hits == MIN_HITS and t.kf.time_since_update == 0 and not t.seen]:
            cand = [o for o in self.tracks if o is not nt and id(o) not in gone and o.seen
                    and o.kf.time_since_update > 0 and o.alive() and (self._close(nt.box, o) if OPTS["diag_close"] else self._claims(nt.box, o))]
            if len(cand) == 1:
                nt.id = cand[0].id; gone.add(id(cand[0]))
        if gone:
            self.tracks = [t for t in self.tracks if id(t) not in gone]
