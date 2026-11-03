"""Compare a trusted original tracker snapshot to the portable package.

Run from the repository root with --snapshot-dir and --halo-dev. No GPU.
The snapshot directory must contain tracker_snapshot.py and legacy_snapshot.py.
"""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exsort.api import create_tracker
from exsort.io import read


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--halo-dev", required=True)
    p.add_argument("--config", default="configs/selected.json")
    args = p.parse_args()
    sys.path.insert(0, args.halo_dev)
    sys.path.insert(0, args.snapshot_dir)
    spec = importlib.util.spec_from_file_location("tracker_snapshot", Path(args.snapshot_dir)/"tracker_snapshot.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    config = read(args.config)
    for k, v in config["params"].items():
        setattr(original, k, v)
    original.OPTS.clear()
    original.OPTS.update(config["fixed_structure"])
    a, b = original.EXSORT(), create_tracker(config)
    rng = np.random.default_rng(20261030)
    compared = 0
    for frame in range(1, 401):
        rows = []
        for i in range(3):
            cx = .15 + i*.30 + .02*np.sin(frame/12+i)
            if rng.random() > .2:
                score = .9 if rng.random() > .2 else .15
                rows.append(dict(box=np.array([cx, .5, .10, .4], np.float32), conf=score))
        mask = np.zeros((512, 512), bool)
        if frame % 17 < 8:
            mask[140:180, 20:480] = True
        out_a = a.step(copy.deepcopy(rows), 2688, 1520, frame, mask)
        out_b = b.step(copy.deepcopy(rows), 2688, 1520, frame, mask)
        if [t.id for t in out_a] != [t.id for t in out_b]:
            raise AssertionError(f"Identity mismatch at frame {frame}")
        for x, y in zip(out_a, out_b):
            np.testing.assert_array_equal(x.out_box(), y.out_box())
            compared += 1
    print(f"PASS: 400 frames, {compared} output boxes, identical IDs and normalized coordinates")


if __name__ == "__main__":
    main()
