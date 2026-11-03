"""Exact FP16 preprocessing parity against the actual YOLOX fork transform."""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exsort.cache import preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolox-root", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.yolox_root)
    from yolox.data.data_augment import ValTransform
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(args.image)
    expected = ValTransform(rgb_means=(.485, .456, .406), std=(.229, .224, .225))(image, None, (800, 1440))[0].astype(np.float16)
    actual = preprocess(image, (800, 1440))
    np.testing.assert_array_equal(actual, expected)
    print("PASS: preprocessed FP16 tensor exactly equals the upstream ValTransform output")


if __name__ == "__main__":
    main()
