"""Ex-Sort: an IoU tracker with glare-explained absence.

Per frame:
1. Keep detections with score >= DET and remove duplicates with NMS.
2. Predict every track with the OC-SORT Kalman filter.
3. Match confirmed tracks, then tentative tracks, with Hungarian IoU matching
   (first against the predicted box, then against the last matched box).
4. A confirmed track that misses a detection is *explained* when the glare mask
   touches the head band of its predicted box. Explained misses do not age the
   track; a track is deleted after more than MAX_AGE unexplained misses.
5. An unmatched detection starts a track only if it is credible (score >= INIT)
   and no existing track claims it.
6. A newly confirmed track takes the ID of the single lost track that claims it.
7. Output: confirmed tracks with a match in this frame, plus confirmed tracks
   under glare for at most GLARE_COAST_SECONDS, if the predicted box still
   overlaps the last matched box (IoU >= EV) and the mean score of the matched
   detections is credible. A track without a credible detection for more than
   MAX_AGE frames gives no output but stays in memory.

A detection is credible when its score is at least INIT. Confirmation needs
MIN_HITS credible matches.
"""
import numpy as np
from .geometry import norm_box_to_xyxy, xyxy_to_norm_box
from .vendor.ocsort.association import iou_batch, linear_assignment
from .vendor.ocsort.ocsort import KalmanBoxTracker

MASK_SIZE = 512  # side of the cached glare mask


def _nms_keep(D, threshold):
    """Greedy NMS by score; returns the kept row indices in ascending order."""
    if len(D) <= 1:
        return list(range(len(D)))
    keep = []
    for i in np.argsort(-D[:, 4]):
        if all(iou_batch(D[i:i + 1, :4], D[j:j + 1, :4])[0, 0] <= threshold for j in keep):
            keep.append(int(i))
    return sorted(keep)


def _iou(a, b):
    return float(iou_batch(np.asarray(a, np.float32)[None], np.asarray(b, np.float32)[None])[0, 0])


def _iom(a, b):
    """Intersection over the smaller box area."""
    a, b = np.asarray(a), np.asarray(b)
    inter = float(np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0.)))
    area = min(float(np.prod(np.maximum(a[2:] - a[:2], 0.))), float(np.prod(np.maximum(b[2:] - b[:2], 0.))))
    return inter / max(area, np.finfo(float).eps)


def _near(b, ref, iou_threshold):
    """Centre distance below rho * longest side, rho = (1 - IOU) / (1 + IOU): the shift at
    which two equal boxes fall out of the matching gate."""
    rho = (1.0 - iou_threshold) / (1.0 + iou_threshold)
    c = 0.5 * (b[:2] + b[2:4]) - 0.5 * (ref[:2] + ref[2:4])
    return float(np.hypot(*c)) < rho * max(b[2] - b[0], b[3] - b[1], ref[2] - ref[0], ref[3] - ref[1], 1.0)


def glare_in_head_band(sat, box, W, H, head_fraction):
    """True if any glare-mask pixel lies in the top head_fraction of the box."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    if x2 - x1 <= 1 or h <= 1:
        return False
    sx, sy = MASK_SIZE / W, MASK_SIZE / H
    a, c = int(max(0, x1 * sx)), int(min(MASK_SIZE, np.ceil(x2 * sx)))
    d, e = int(max(0, y1 * sy)), int(min(MASK_SIZE, np.ceil((y1 + head_fraction * h) * sy)))
    if c <= a or e <= d:
        return False
    return bool(sat[d:e, a:c].any())


class Track:
    def __init__(self, tid, row, W, H):
        self.id = tid
        self.kf = KalmanBoxTracker(row[:4])
        self.kf.update(np.asarray(row[:5], np.float32).copy())
        self.W, self.H = W, H
        self.box = row[:4].copy()      # last matched detection
        self.credible_hits = 1         # births are credible
        self.since_credible = 0        # frames since the last credible match
        self.score_sum, self.score_n = float(row[4]), 1
        self.unexplained = 0           # misses not explained by glare
        self.glare = False             # miss explained by glare in this frame
        self.seen = False              # has been confirmed

    def pred(self):
        return self.kf.get_state()[0][:4].astype(np.float32)

    def out_box(self):
        """Kalman box: the posterior after a match, the prediction otherwise."""
        b = self.kf.get_state()[0][:4]
        if self.kf.time_since_update == 0:
            b = b.astype(np.float32)
        return xyxy_to_norm_box(b, self.W, self.H)

    def observe(self, row, credible):
        self.kf.update(np.asarray(row[:5], np.float32).copy())
        self.box = row[:4].copy()
        self.unexplained, self.glare = 0, False
        self.score_sum += float(row[4])
        self.score_n += 1
        if row[4] >= credible:
            self.since_credible = 0
            self.credible_hits += 1


class ExSort:
    def __init__(self, fps, *, DET, INIT, NMS, IOU, MIN_HITS, MAX_AGE, HEAD_FRACTION, EV, GLARE_COAST_SECONDS):
        self.fps = fps
        self.det, self.init, self.nms, self.iou = DET, INIT, NMS, IOU
        self.min_hits, self.max_age = MIN_HITS, MAX_AGE
        self.head_fraction, self.ev = HEAD_FRACTION, EV
        self.glare_coast_frames = GLARE_COAST_SECONDS * fps
        self.tracks, self.n, self.frame = [], 0, 0

    def confirmed(self, t):
        return t.credible_hits >= self.min_hits

    def _claims(self, b, t):
        """Whether existing track t covers box b (used to block births and to recover IDs)."""
        if t.kf.time_since_update > 0:
            return self._lost_claims(b, t)
        return _iom(b, t.box) >= self.iou

    def _lost_claims(self, b, t):
        """A lost track claims b when b is near its predicted or last matched box."""
        return _near(b, t.pred(), self.iou) or _near(b, t.box, self.iou)

    def _match(self, D, ud, ut, P):
        """Two rounds of Hungarian IoU matching: predicted boxes, then last matched boxes."""
        for use_last in (False, True):
            if not ud or not ut:
                break
            refs = np.asarray([self.tracks[i].box if use_last else P[i] for i in ut], np.float32)
            iou = iou_batch(D[ud, :4], refs)
            pairs = [] if not (iou >= self.iou).any() else \
                [(int(d), int(t)) for d, t in linear_assignment(-iou) if iou[d, t] >= self.iou]
            for d, t in pairs:
                self.tracks[ut[t]].observe(D[ud[d]], self.init)
            md, mt = {d for d, _ in pairs}, {t for _, t in pairs}
            ud = [d for i, d in enumerate(ud) if i not in md]
            ut = [t for i, t in enumerate(ut) if i not in mt]
        return ud, ut

    def _coasts(self, t):
        return (t.glare and t.score_sum / t.score_n >= self.init
                and _iou(t.pred(), t.box) >= self.ev
                and t.kf.time_since_update <= self.glare_coast_frames)

    def _recover(self):
        """A track confirmed in this frame takes the ID of the one lost track that claims it."""
        if self.frame <= self.min_hits:
            return
        gone = set()
        for nt in [t for t in self.tracks if t.kf.hits == self.min_hits and t.kf.time_since_update == 0 and not t.seen]:
            cand = [o for o in self.tracks if o is not nt and id(o) not in gone and o.seen
                    and o.kf.time_since_update > 0 and self._lost_claims(nt.box, o)]
            if len(cand) == 1:
                nt.id = cand[0].id
                gone.add(id(cand[0]))
        if gone:
            self.tracks = [t for t in self.tracks if id(t) not in gone]

    def step(self, dets, W, H, sat=None):
        """dets: list of dict(box=[cx, cy, w, h] normalised, conf=score). sat: 512x512 glare mask."""
        self.frame += 1
        D = np.asarray([list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']] for d in dets if d['conf'] >= self.det],
                       np.float32).reshape(-1, 5)
        D = D[_nms_keep(D, self.nms)]
        for t in self.tracks:
            t.since_credible += 1
        P = np.asarray([t.kf.predict()[0][:4] for t in self.tracks], np.float32).reshape(-1, 4)

        ud = list(range(len(D)))
        confirmed = [i for i, t in enumerate(self.tracks) if self.confirmed(t)]
        tentative = [i for i, t in enumerate(self.tracks) if not self.confirmed(t)]
        ud, missed = self._match(D, ud, confirmed, P)
        ud, missed_tentative = self._match(D, ud, tentative, P)

        for i in missed + missed_tentative:
            t = self.tracks[i]
            t.kf.update(None)
            t.glare = (sat is not None and self.confirmed(t)
                       and glare_in_head_band(sat, t.pred(), W, H, self.head_fraction))
            if not t.glare:
                t.unexplained += 1

        # A tentative track is dropped at its first miss; any track ages out on unexplained misses.
        self.tracks = [t for t in self.tracks if (self.confirmed(t) or t.kf.time_since_update == 0)
                       and t.unexplained <= self.max_age]

        for d in ud:
            row = D[d]
            if row[4] < self.init or any(self._claims(row[:4], t) for t in self.tracks):
                continue
            self.n += 1
            self.tracks.append(Track(self.n, row, W, H))

        self._recover()
        out = [t for t in self.tracks if self.confirmed(t) and t.since_credible <= self.max_age
               and (t.kf.time_since_update == 0 or self._coasts(t))]
        for t in self.tracks:
            if self.confirmed(t):
                t.seen = True
        return out
