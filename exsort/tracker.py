"""Ex-Sort: an IoU tracker with glare-explained absence.

A detection is *credible* when its score is at least INIT. For each frame:

1. Keep detections with score >= DET and remove duplicates with NMS.
2. Predict every track with the OC-SORT Kalman filter.
3. Match confirmed tracks, then tentative tracks, with Hungarian IoU matching:
   first against the predicted box, then against the last matched box.
4. A tentative track that misses a detection is deleted. A confirmed track
   that misses a detection is *explained* when the glare mask touches the head
   band of its predicted box. Explained misses do not age the track; a track is
   deleted after more than MAX_AGE unexplained misses.
5. An unmatched credible detection starts a track if no existing track claims it.
   A track is confirmed after MIN_HITS credible matches.
6. A new track with exactly MIN_HITS matches, matched in this frame, takes the
   ID of a lost track if exactly one lost track claims it.
7. Output confirmed tracks that were matched in this frame. A confirmed track
   that missed is output with its predicted box only under glare, for at most
   GLARE_COAST_SECONDS, if the predicted box still overlaps the last matched box
   (IoU >= EV) and the mean score of its matched detections is credible.
   A track whose last credible detection is more than MAX_AGE frames old gives
   no output but stays in memory.

Boxes inside the tracker are pixel [x1, y1, x2, y2].
"""
import numpy as np
from .geometry import norm_box_to_xyxy, xyxy_to_norm_box
from .vendor.ocsort.association import iou_batch, linear_assignment
from .vendor.ocsort.ocsort import KalmanBoxTracker

MASK_SIZE = 512  # side of the cached glare mask


def nms_keep(dets, threshold):
    """Greedy NMS by score. Returns the indices of the kept rows in ascending order."""
    if len(dets) <= 1:
        return list(range(len(dets)))
    keep = []
    for i in np.argsort(-dets[:, 4]):
        if all(iou_batch(dets[i:i + 1, :4], dets[j:j + 1, :4])[0, 0] <= threshold for j in keep):
            keep.append(int(i))
    return sorted(keep)


def iou(a, b):
    return float(iou_batch(np.asarray(a, np.float32)[None], np.asarray(b, np.float32)[None])[0, 0])


def iom(a, b):
    """Intersection over the area of the smaller box."""
    a, b = np.asarray(a), np.asarray(b)
    inter = float(np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0.)))
    area = min(float(np.prod(np.maximum(a[2:] - a[:2], 0.))), float(np.prod(np.maximum(b[2:] - b[:2], 0.))))
    return inter / max(area, np.finfo(float).eps)


def near(box, ref, iou_threshold):
    """Axis gate: the centre distance is below rho * the longest side, with
    rho = (1 - IOU) / (1 + IOU), the shift at which two equal boxes fall below the IoU gate."""
    rho = (1.0 - iou_threshold) / (1.0 + iou_threshold)
    shift = 0.5 * (box[:2] + box[2:4]) - 0.5 * (ref[:2] + ref[2:4])
    longest = max(box[2] - box[0], box[3] - box[1], ref[2] - ref[0], ref[3] - ref[1], 1.0)
    return float(np.hypot(*shift)) < rho * longest


def glare_in_head_band(mask, box, W, H, head_fraction):
    """True if any glare-mask pixel lies in the top head_fraction of the box."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    if x2 - x1 <= 1 or height <= 1:
        return False
    sx, sy = MASK_SIZE / W, MASK_SIZE / H
    left, right = int(max(0, x1 * sx)), int(min(MASK_SIZE, np.ceil(x2 * sx)))
    top, bottom = int(max(0, y1 * sy)), int(min(MASK_SIZE, np.ceil((y1 + head_fraction * height) * sy)))
    if right <= left or bottom <= top:
        return False
    return bool(mask[top:bottom, left:right].any())


class Track:
    def __init__(self, track_id, det, W, H):
        self.id = track_id
        self.kf = KalmanBoxTracker(det[:4])
        self.kf.update(np.asarray(det[:5], np.float32).copy())
        self.W, self.H = W, H
        self.box = det[:4].copy()      # last matched detection
        self.credible_hits = 1         # a track starts from a credible detection
        self.since_credible = 0        # frames since the last credible match
        self.score_sum, self.score_count = float(det[4]), 1
        self.unexplained = 0           # consecutive misses not explained by glare
        self.glare = False             # this frame's miss is explained by glare
        self.seen = False              # was confirmed in an earlier frame

    def missed(self):
        return self.kf.time_since_update > 0

    def predicted_box(self):
        return self.kf.get_state()[0][:4].astype(np.float32)

    def out_box(self):
        """Kalman box (posterior after a match, prediction after a miss) as normalised [cx, cy, w, h].
        The float32 cast for matched tracks reproduces the published results digit for digit."""
        box = self.kf.get_state()[0][:4]
        if not self.missed():
            box = box.astype(np.float32)
        return xyxy_to_norm_box(box, self.W, self.H)

    def update(self, det, credible_score):
        self.kf.update(np.asarray(det[:5], np.float32).copy())
        self.box = det[:4].copy()
        self.unexplained, self.glare = 0, False
        self.score_sum += float(det[4])
        self.score_count += 1
        if det[4] >= credible_score:
            self.since_credible = 0
            self.credible_hits += 1


class ExSort:
    def __init__(self, fps, *, DET, INIT, NMS, IOU, MIN_HITS, MAX_AGE, HEAD_FRACTION, EV, GLARE_COAST_SECONDS):
        self.det_score, self.credible_score = DET, INIT
        self.nms, self.iou = NMS, IOU
        self.min_hits, self.max_age = MIN_HITS, MAX_AGE
        self.head_fraction, self.ev = HEAD_FRACTION, EV
        self.glare_coast_frames = GLARE_COAST_SECONDS * fps
        self.tracks, self.last_id, self.frame = [], 0, 0

    def confirmed(self, track):
        return track.credible_hits >= self.min_hits

    def lost_track_claims(self, box, track):
        """A lost track claims a box near its predicted or last matched box."""
        return near(box, track.predicted_box(), self.iou) or near(box, track.box, self.iou)

    def claims(self, box, track):
        """Whether an existing track covers box, so that no new track starts there."""
        if track.missed():
            return self.lost_track_claims(box, track)
        return iom(box, track.box) >= self.iou

    def match(self, dets, free_dets, track_idx, predicted):
        """Hungarian IoU matching in two rounds: predicted boxes, then last matched boxes.
        Returns the detections and tracks that are still unmatched."""
        for use_last_box in (False, True):
            if not free_dets or not track_idx:
                break
            refs = np.asarray([self.tracks[i].box if use_last_box else predicted[i] for i in track_idx], np.float32)
            overlap = iou_batch(dets[free_dets, :4], refs)
            pairs = []
            if (overlap >= self.iou).any():
                pairs = [(int(d), int(t)) for d, t in linear_assignment(-overlap) if overlap[d, t] >= self.iou]
            for d, t in pairs:
                self.tracks[track_idx[t]].update(dets[free_dets[d]], self.credible_score)
            matched_d, matched_t = {d for d, _ in pairs}, {t for _, t in pairs}
            free_dets = [x for k, x in enumerate(free_dets) if k not in matched_d]
            track_idx = [x for k, x in enumerate(track_idx) if k not in matched_t]
        return free_dets, track_idx

    def coasts(self, track):
        """Whether a confirmed track that missed is output with its predicted box."""
        return (track.glare
                and track.score_sum / track.score_count >= self.credible_score
                and iou(track.predicted_box(), track.box) >= self.ev
                and track.kf.time_since_update <= self.glare_coast_frames)

    def recover_ids(self):
        """A new track with exactly MIN_HITS matches, matched in this frame, takes the ID
        of a lost confirmed track if exactly one such track claims it."""
        if self.frame <= self.min_hits:
            return
        replaced = set()
        new_tracks = [t for t in self.tracks if t.kf.hits == self.min_hits and not t.missed() and not t.seen]
        for new in new_tracks:
            candidates = [old for old in self.tracks if old is not new and id(old) not in replaced
                          and old.seen and old.missed() and self.lost_track_claims(new.box, old)]
            if len(candidates) == 1:
                new.id = candidates[0].id
                replaced.add(id(candidates[0]))
        if replaced:
            self.tracks = [t for t in self.tracks if id(t) not in replaced]

    def step(self, dets, W, H, sat=None):
        """Process one frame and return the tracks to output.
        dets: list of dict(box=[cx, cy, w, h] normalised by W and H, conf=score).
        sat: 512 x 512 boolean glare mask of this frame."""
        self.frame += 1

        # 1. Detections with score >= DET, then NMS.
        rows = [list(norm_box_to_xyxy(d['box'], W, H)) + [d['conf']] for d in dets if d['conf'] >= self.det_score]
        dets = np.asarray(rows, np.float32).reshape(-1, 5)
        dets = dets[nms_keep(dets, self.nms)]

        # 2. Kalman prediction.
        for t in self.tracks:
            t.since_credible += 1
        predicted = np.asarray([t.kf.predict()[0][:4] for t in self.tracks], np.float32).reshape(-1, 4)

        # 3. Match confirmed tracks first, then tentative tracks.
        free_dets = list(range(len(dets)))
        confirmed = [i for i, t in enumerate(self.tracks) if self.confirmed(t)]
        tentative = [i for i, t in enumerate(self.tracks) if not self.confirmed(t)]
        free_dets, missed_confirmed = self.match(dets, free_dets, confirmed, predicted)
        free_dets, missed_tentative = self.match(dets, free_dets, tentative, predicted)

        # 4. Misses: glare in the head band explains the miss of a confirmed track.
        for i in missed_confirmed + missed_tentative:
            t = self.tracks[i]
            t.kf.update(None)
            t.glare = (sat is not None and self.confirmed(t)
                       and glare_in_head_band(sat, t.predicted_box(), W, H, self.head_fraction))
            if not t.glare:
                t.unexplained += 1
        self.tracks = [t for t in self.tracks
                       if (self.confirmed(t) or not t.missed()) and t.unexplained <= self.max_age]

        # 5. New tracks from credible detections that no track claims.
        for d in free_dets:
            det = dets[d]
            if det[4] < self.credible_score or any(self.claims(det[:4], t) for t in self.tracks):
                continue
            self.last_id += 1
            self.tracks.append(Track(self.last_id, det, W, H))

        # 6. ID recovery.
        self.recover_ids()

        # 7. Output.
        out = [t for t in self.tracks if self.confirmed(t) and t.since_credible <= self.max_age
               and (not t.missed() or self.coasts(t))]
        for t in self.tracks:
            if self.confirmed(t):
                t.seen = True
        return out
