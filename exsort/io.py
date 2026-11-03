"""Atomic metadata and reproducibility fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(native(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_signature():
    root = Path(__file__).parent
    return key({f.relative_to(root).as_posix(): sha(f) for f in sorted(root.rglob("*.py"))})
