"""CPU replay, resumable per-block tracking, cached per-sequence TrackEval."""
import gzip
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
from .api import create_tracker, validate
from .cache import open_cache, verify
from .dataset import current_gt
from .geometry import norm_box_to_xyxy
from .io import key, read, save, sha, source_signature
from . import scoring


def track_one(job):
    manifest, cache, config, output, name, track_identity = job
    cache, output = Path(cache), Path(output)
    q = manifest["sequences"][name]
    folder = output / "work" / name
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "done.json"
    target = output / "trackers/exsort/data" / (name + ".txt")
    if done.exists():
        d = read(done)
        if d["identity"] != track_identity or sha(target) != d["sha256"]:
            raise ValueError("Completed tracking result changed")
        return d
    model = create_tracker(config)
    checkpoint = folder / "checkpoint.pkl.gz"
    current, position = 0, 0
    if checkpoint.exists():
        with gzip.open(checkpoint, "rb") as f:
            saved = pickle.load(f)
        if saved["identity"] != track_identity:
            raise ValueError("Checkpoint protocol changed; use a fresh run")
        model, current, position = saved["model"], saved["frame"], saved["position"]
    partial = folder / "predictions.partial.txt"
    began = time.perf_counter()
    with partial.open("r+b" if partial.exists() else "w+b") as f:
        f.seek(0, 2)
        if f.tell() < position:
            raise ValueError("Prediction partial file is shorter than checkpoint")
        f.truncate(position)
        f.seek(position)
        for b in read(cache / name / "index.json")["blocks"]:
            if b["end"] <= current:
                continue
            path = cache / name / b["file"]
            if sha(path) != b["sha256"]:
                raise ValueError("Cache block changed")
            with np.load(path, allow_pickle=False) as z:
                off, det, masks = z["offsets"], z["detections"], z["masks"]
            for k in range(max(current, b["start"]), b["end"]):
                i = k - b["start"]
                rows = []
                for r in det[off[i]:off[i+1]]:
                    if r[4] < min(config["params"]["DET"], config["params"]["LOW"]):
                        continue
                    x1, y1, x2, y2 = r[:4]
                    rows.append(dict(box=np.array([(x1+x2)/2/q["width"], (y1+y2)/2/q["height"],
                                                   (x2-x1)/q["width"], (y2-y1)/q["height"]], np.float32), conf=float(r[4])))
                sat = np.unpackbits(masks[i]).reshape(512, 512).astype(bool)
                tracks = model.step(rows, q["width"], q["height"], k+1, sat=sat)
                lines = []
                decimals = 2 if manifest["role"] == "val" else 4
                for tr in tracks:
                    x1, y1, x2, y2 = norm_box_to_xyxy(tr.out_box(), q["width"], q["height"])
                    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                        raise ValueError("Invalid tracker output")
                    values = (max(0., x1), max(0., y1), max(1., x2-x1), max(1., y2-y1))
                    lines.append(f"{k+1},{tr.id}," + ",".join(f"{v:.{decimals}f}" for v in values) + ",1,-1,-1,-1\n")
                f.write("".join(lines).encode("ascii"))
                current = k+1
            f.flush()
            os.fsync(f.fileno())
            tmp = folder / "checkpoint.tmp.gz"
            with gzip.open(tmp, "wb", compresslevel=1) as cf:
                pickle.dump(dict(identity=track_identity, model=model, frame=current, position=f.tell()), cf, pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, checkpoint)
            save(folder / "progress.json", dict(frame=current, frames=q["frames"]))
            if (output / "STOP").exists():
                return dict(sequence=name, paused=True, frames=current)
    if current != q["frames"]:
        raise ValueError("Incomplete cache replay")
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    tmp = target.with_suffix(".tmp")
    shutil.copyfile(partial, tmp)
    os.replace(tmp, target)
    record = dict(sequence=name, frames=current, identity=track_identity, sha256=sha(target), seconds=time.perf_counter()-began)
    save(done, record)
    return record


def score_one(job):
    manifest, output, name = job
    output = Path(output)
    q = manifest["sequences"][name]
    gt = current_gt(manifest, name)
    target = output / "trackers/exsort/data" / (name + ".txt")
    signature = key(dict(gt=q["gt_sha256"], prediction=sha(target), scoring=sha(scoring.__file__),
                         trackeval_version=trackeval_version()))
    dest = output / "scores" / (name+".pkl")
    if dest.exists():
        with dest.open("rb") as f:
            old = pickle.load(f)
        if old["signature"] != signature:
            raise ValueError("Scoring inputs changed; create a new scoring run")
        return name, old["metrics"]
    pred = np.loadtxt(target, delimiter=",", ndmin=2)[:, :6] if target.stat().st_size else np.empty((0, 6))
    result = scoring.evaluate_sequence(gt, pred, q["frames"], 2 if manifest["role"] == "val" else 4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    with temp.open("wb") as f:
        pickle.dump(dict(signature=signature, metrics=result), f, pickle.HIGHEST_PROTOCOL)
    os.replace(temp, dest)
    return name, result


def trackeval_version():
    import importlib.metadata as metadata
    return metadata.version("trackeval")


def worker_manifest(manifest, name):
    # Avoid pickling every image filename/fingerprint for every sequence/trial.
    q = manifest["sequences"][name]
    return dict(data_root=manifest["data_root"], role=manifest["role"],
                sequences={name: {k: v for k, v in q.items() if k not in ("images", "image_fingerprint")}})


def replay(manifest, cache, config, output, workers=4, pool=None, validated=False):
    config = validate(config)
    output = Path(output)
    meta = open_cache(cache, manifest) if validated else verify(cache, manifest)
    if config["params"]["LOW"] < meta["score_floor"] or config["params"]["NMS"] > meta["nms"]:
        raise ValueError("Config requires detections removed by the cache")
    blocks = {s: read(Path(cache)/s/"index.json")["blocks"] for s in manifest["sequences"]}
    track_identity = key(dict(cache=meta["identity"], blocks=blocks, config=config, source=source_signature(),
                              decimals=2 if manifest["role"] == "val" else 4))
    protocol = dict(manifest_signature=manifest["signature"], role=manifest["role"],
                    cache_identity=meta["identity"], config=config, source_signature=source_signature(),
                    track_identity=track_identity, trackeval_version=trackeval_version())
    pp = output / "protocol.json"
    if pp.exists() and read(pp) != protocol:
        raise ValueError("Run inputs changed. Use a new output directory.")
    save(pp, protocol)
    own_pool = pool is None
    pool = pool or ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        jobs = [(worker_manifest(manifest, s), str(cache), config, str(output), s, track_identity)
                for s in sorted(manifest["sequences"], key=lambda s: -manifest["sequences"][s]["frames"])]
        tracked = [f.result() for f in as_completed([pool.submit(track_one, j) for j in jobs])]
        if any(r.get("paused") for r in tracked):
            return None
        per_seq = dict(f.result() for f in as_completed([pool.submit(score_one, (worker_manifest(manifest, s), str(output), s)) for s in manifest["sequences"]]))
        combined = scoring.combine(per_seq)
        result = dict(config=config, summary=scoring.summary(combined),
                      per_sequence={s: scoring.summary(r) for s, r in per_seq.items()},
                      track_identity=track_identity, frames=manifest["frames"], sequences=len(per_seq))
        save(output / "all_trackeval_metrics.json", dict(combined=combined, per_sequence=per_seq))
        save(output / "metrics.json", result)
        save(output / "complete.json", dict(completed=True, **result))
        return result
    finally:
        if own_pool:
            pool.shutdown(wait=True, cancel_futures=True)


def freeze(config, manifest, cache, metrics, path, origin="validation-selected"):
    config = validate(config)
    if manifest["role"] != "val":
        raise ValueError("Selection can only be frozen from val")
    meta = open_cache(cache, manifest)
    selection = dict(schema=1, config=config, config_sha256=key(config),
                     selected_on=origin, validation_manifest=manifest["signature"],
                     split_signature=manifest["split_signature"], train_sequences=manifest["split"]["train"],
                     source_signature=source_signature(), validation_metrics=metrics,
                     detector_sha256=meta.get("detector_sha256", meta.get("provenance", {}).get("detector_sha256")))
    selection["seal"] = key(selection)
    path = Path(path)
    if path.exists() and read(path) != selection:
        raise ValueError("Selection is already frozen. Use a new selection path.")
    save(path, selection)
    return selection


def evaluate_test(args):
    manifest, selection = read(args.manifest), read(args.selection)
    expected = dict(selection)
    seal = expected.pop("seal")
    if key(expected) != seal or key(selection["config"]) != selection["config_sha256"]:
        raise ValueError("Frozen selection was modified")
    if manifest["role"] != "test" or manifest["split_signature"] != selection["split_signature"]:
        raise ValueError("Test split differs from frozen validation protocol")
    if selection["source_signature"] != source_signature():
        raise ValueError("Source changed after selection was frozen")
    meta = open_cache(args.cache, manifest)
    test_model = meta.get("detector_sha256", meta.get("provenance", {}).get("detector_sha256"))
    if not test_model or test_model != selection["detector_sha256"]:
        raise ValueError("Use the same identified detector checkpoint for val and test")
    result = replay(manifest, args.cache, selection["config"], args.output, args.workers)
    if result:
        save(Path(args.output) / "frozen_selection.json", selection)
        print(result["summary"], flush=True)


def rescore(args):
    """New GT, same images: reuse MOT predictions, no detector/tracker replay."""
    import shutil
    manifest = read(args.manifest)
    source, out = Path(args.predictions), Path(args.output)
    # Require the originating manifest to prove the image timeline is unchanged.
    original = read(args.original_manifest)
    from .cache import image_identity
    if image_identity(original) != image_identity(manifest):
        raise ValueError("Images/timeline changed; predictions cannot be reused")
    for s in manifest["sequences"]:
        src = source / (s+".txt")
        dst = out / "trackers/exsort/data" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(src) != sha(dst):
            raise ValueError("Existing predictions differ")
        if not dst.exists():
            shutil.copyfile(src, dst)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        per_seq = dict(pool.map(score_one, [(worker_manifest(manifest, s), str(out), s) for s in manifest["sequences"]]))
    combined = scoring.combine(per_seq)
    save(out/"all_trackeval_metrics.json", dict(combined=combined, per_sequence=per_seq))
    save(out/"metrics.json", dict(summary=scoring.summary(combined), per_sequence={s: scoring.summary(x) for s, x in per_seq.items()},
         GT_manifest=manifest["signature"], original_image_identity=image_identity(original)))
    print(scoring.summary(combined), flush=True)
