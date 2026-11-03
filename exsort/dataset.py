"""Dense, sequence-disjoint train/test manifests with temporal tracker validation."""
from pathlib import Path
import cv2
import numpy as np
from .io import key, read, save, sha


def load_gt(path, nframes):
    if Path(path).stat().st_size == 0:
        return np.empty((0, 9), dtype=np.float64)
    g = np.loadtxt(path, delimiter=",", ndmin=2)
    if g.shape[1] != 9 or not np.isfinite(g).all():
        raise ValueError(f"GT must have 9 finite MOT columns: {path}")
    if (g[:, 4:6] <= 0).any() or (g[:, 7] != 1).any():
        raise ValueError(f"Only person class 1 and positive boxes supported: {path}")
    if (g[:, 0] < 1).any() or (g[:, 0] > nframes).any() or not np.equal(g[:, :2], np.floor(g[:, :2])).all():
        raise ValueError(f"Invalid frame or identity: {path}")
    if (g[:, 1] < 0).any() or len(np.unique(g[:, :2], axis=0)) != len(g):
        raise ValueError(f"Duplicate frame/ID or negative ID: {path}")
    return g


def split_names(split):
    result = {}
    for name in ("train", "test"):
        result[name] = [x if isinstance(x, str) else x["sequence"] for x in split[name]]
        if len(result[name]) != len(set(result[name])):
            raise ValueError("Duplicate sequence in split")
        for s in result[name]:
            if Path(s).name != s or "/" in s or "\\" in s or s in (".", ".."):
                raise ValueError("Unsafe sequence name")
    if set(result["train"]) & set(result["test"]):
        raise ValueError("Train/test sequence overlap")
    return result


def prepare(data, split_path, role, output, full_hash=False):
    """val = full dense second half of each train sequence, no sampling/filtering."""
    data = Path(data).resolve()
    split = read(split_path)
    names = split_names(split)
    physical = "train" if role == "val" else "test"
    sequences = {}
    for s in names[physical]:
        folder = data / physical / s
        images = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not images:
            raise ValueError(f"No images: {folder}")
        # A missing frame must not silently shift the MOT timeline.
        numbers = [int(p.stem.removeprefix("frame_")) for p in images]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"Non-contiguous image names: {folder}")
        first = len(images) // 2 if role == "val" else 0
        selected = images[first:]
        image = cv2.imread(str(selected[0]))
        if image is None:
            raise ValueError(str(selected[0]))
        h, w = image.shape[:2]
        records = []
        for p in selected:
            st = p.stat()
            r = dict(name=p.name, bytes=st.st_size, mtime_ns=st.st_mtime_ns)
            if full_hash:
                r["sha256"] = sha(p)
            records.append(r)
        # Default fast mode also hashes the first and last selected image contents.
        fingerprint = dict(mode="sha256-all" if full_hash else "names-size-mtime-and-endpoint-sha256",
                           entries=records, endpoints=[sha(selected[0]), sha(selected[-1])])
        gt = folder / "mot/gt/gt.txt"
        load_gt(gt, len(images))
        sequences[s] = dict(split=physical, n_total=len(images), start=first, frames=len(selected),
                            width=w, height=h, images=[p.name for p in selected],
                            image_fingerprint=fingerprint, gt_sha256=sha(gt))
        print(f"Indexed {role}: {s} ({len(selected)} frames)", flush=True)
    manifest = dict(schema=1, role=role, data_root=str(data), split=names,
                    split_signature=key(names), sequences=sequences,
                    frames=sum(q["frames"] for q in sequences.values()),
                    policy="dense second half of train; tracker starts fresh" if role == "val" else "dense full held-out test sequences")
    # Dataset root is relocatable; all identities describe contents, not drive letters.
    manifest["signature"] = key({k: v for k, v in manifest.items() if k != "data_root"})
    output = Path(output)
    if output.exists() and read(output) != manifest:
        raise ValueError("Manifest already exists with different data. Use a new path.")
    save(output, manifest)
    return manifest


def assert_images(manifest, name):
    q = manifest["sequences"][name]
    root = Path(manifest["data_root"]) / q["split"] / name / "images"
    for r in q["image_fingerprint"]["entries"]:
        p = root / r["name"]
        st = p.stat()
        if (st.st_size, st.st_mtime_ns) != (r["bytes"], r["mtime_ns"]):
            raise ValueError(f"Image changed; prepare a new manifest/cache: {p}")
        if "sha256" in r and sha(p) != r["sha256"]:
            raise ValueError(f"Image content changed: {p}")
    for filename, expected in zip((q["images"][0], q["images"][-1]), q["image_fingerprint"]["endpoints"]):
        if sha(root / filename) != expected:
            raise ValueError(f"Image endpoint changed: {name}")


def current_gt(manifest, name):
    q = manifest["sequences"][name]
    path = Path(manifest["data_root"]) / q["split"] / name / "mot/gt/gt.txt"
    if sha(path) != q["gt_sha256"]:
        raise ValueError(f"GT changed since manifest: {name}. Prepare a new manifest and scoring run.")
    gt = load_gt(path, q["n_total"])
    gt = gt[gt[:, 0] > q["start"]].copy()
    gt[:, 0] -= q["start"]
    if manifest["role"] == "val":
        # Historical validation GT was written with two decimal places.
        gt[:, 2:6] = np.round(gt[:, 2:6], 2)
    return gt
