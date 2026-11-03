"""TrackEval HOTA/CLEAR/Identity/Count, with grouped in-memory MOT data."""
import numpy as np

# TrackEval's pinned historical commit predates NumPy 2.
for _name, _value in {"float": float, "int": int, "bool": bool, "float_": np.float64}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


def metrics():
    import trackeval
    opts = dict(THRESHOLD=.5, PRINT_CONFIG=False)
    return [trackeval.metrics.HOTA(opts), trackeval.metrics.CLEAR(opts),
            trackeval.metrics.Identity(opts), trackeval.metrics.Count()]


def grouped(gt, predictions, frames, decimals):
    from trackeval.datasets._base_dataset import _BaseDataset
    p = np.asarray(predictions, np.float64).reshape(-1, 6).copy()
    p[:, 2:6] = np.round(p[:, 2:6], decimals)
    gt = np.asarray(gt, np.float64).reshape(-1, 9)
    gt = gt[gt[:, 6] != 0]
    result = {k: [] for k in ("gt_ids", "tracker_ids", "gt_dets", "tracker_dets", "similarity_scores")}
    bounds = []
    for a in (gt, p):
        if len(a) and (not np.isfinite(a).all() or (a[:, 0] < 1).any() or (a[:, 0] > frames).any()):
            raise ValueError("Invalid MOT data")
        if len(np.unique(a[:, :2], axis=0)) != len(a):
            raise ValueError("Duplicate frame/identity")
        a[:] = a[np.argsort(a[:, 0], kind="stable")]
        bounds.append(np.searchsorted(a[:, 0], np.arange(1, frames + 2)))
    bg, bp = bounds
    for k in range(frames):
        g, t = gt[bg[k]:bg[k+1]], p[bp[k]:bp[k+1]]
        gi, ti = g[:, 1].astype(int), t[:, 1].astype(int)
        gb, tb = g[:, 2:6], t[:, 2:6]
        result["gt_ids"].append(gi)
        result["tracker_ids"].append(ti)
        result["gt_dets"].append(gb)
        result["tracker_dets"].append(tb)
        result["similarity_scores"].append(_BaseDataset._calculate_box_ious(gb, tb, box_format="xywh"))
    for prefix in ("gt", "tracker"):
        ids = result[prefix + "_ids"]
        unique = np.unique(np.concatenate(ids)) if ids else np.empty(0, int)
        result[prefix + "_ids"] = [np.searchsorted(unique, a).astype(int) for a in ids]
        result["num_" + prefix + "_ids"] = len(unique)
        result["num_" + prefix + "_dets"] = sum(len(a) for a in ids)
    result["num_timesteps"] = frames
    return result


def evaluate_sequence(gt, predictions, frames, decimals=4):
    data = grouped(gt, predictions, frames, decimals)
    return {m.get_name(): m.eval_sequence(data) for m in metrics()}


def combine(rows):
    return {m.get_name(): m.combine_sequences({s: r[m.get_name()] for s, r in rows.items()}) for m in metrics()}


def summary(r):
    h, c, i = r["HOTA"], r["CLEAR"], r["Identity"]
    return dict(HOTA=100 * float(np.mean(h["HOTA"])), DetA=100 * float(np.mean(h["DetA"])),
                AssA=100 * float(np.mean(h["AssA"])), IDF1=100 * float(i["IDF1"]),
                MOTA=100 * float(c["MOTA"]), IDSW=int(c["IDSW"]), FP=int(c["CLR_FP"]), FN=int(c["CLR_FN"]))
