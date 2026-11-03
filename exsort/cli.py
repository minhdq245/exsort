"""Command-line entry point. See README.md for the complete protocol."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
from pathlib import Path


def manifest_cache(p):
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache", required=True)


def main():
    from .locking import writer_lock
    from contextlib import nullcontext
    args = parse_args()
    # CPU workers do not call this entry point and do not acquire the parent lock.
    output = getattr(args, "output", None)
    folder = (Path(output).parent if args.command == "prepare" else output) if output else None
    with writer_lock(folder) if folder else nullcontext():
        dispatch(args)


def parse_args():
    p = argparse.ArgumentParser(description="exsort: cache -> validation tune -> frozen test evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Snapshot dense image timeline and current GT")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--role", choices=["val", "test"], required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--hash-images", action="store_true", help="SHA256 every image (more disk I/O)")
    cache = sub.add_parser("cache", help="YOLOX person detections and measured saturation masks, once")
    cache.add_argument("--manifest", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--yolox-root", required=True)
    cache.add_argument("--exp", default="examples/yolox_x_person.py")
    cache.add_argument("--weights", required=True)
    cache.add_argument("--height", type=int, default=800)
    cache.add_argument("--width", type=int, default=1440)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    cache.add_argument("--batch-size", type=int, default=4)
    cache.add_argument("--workers", type=int, default=4)
    cache.add_argument("--block-frames", type=int, default=512)
    imp = sub.add_parser("import-cache", help="Import existing trusted research caches without GPU inference")
    imp.add_argument("--manifest", required=True)
    imp.add_argument("--source", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--format", required=True, choices=["val-mm", "test-blocks"])
    imp.add_argument("--mask-map")
    imp.add_argument("--detector-sha256", help="Required provenance for a legacy val cache")
    imp.add_argument("--trust-legacy-images", action="store_true")
    check = sub.add_parser("verify-cache")
    manifest_cache(check)
    val = sub.add_parser("validate", help="Replay one config on dense val and freeze it")
    manifest_cache(val)
    val.add_argument("--config", default="configs/selected.json")
    val.add_argument("--output", required=True)
    val.add_argument("--workers", type=int, default=4)
    tune = sub.add_parser("tune", help="Sequential SMAC Bayesian optimization, val only")
    manifest_cache(tune)
    tune.add_argument("--initial-config", default="configs/selected.json")
    tune.add_argument("--baseline", default="configs/baseline.json")
    tune.add_argument("--search", default="configs/search_local.json")
    tune.add_argument("--output", required=True)
    tune.add_argument("--workers", type=int, default=4)
    abl = sub.add_parser("ablate")
    manifest_cache(abl)
    abl.add_argument("--config", default="configs/selected.json")
    abl.add_argument("--variants", default="configs/ablations.json")
    abl.add_argument("--output", required=True)
    abl.add_argument("--workers", type=int, default=4)
    test = sub.add_parser("evaluate", help="One frozen selection on dense full test")
    manifest_cache(test)
    test.add_argument("--selection", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--workers", type=int, default=4)
    score = sub.add_parser("rescore", help="New GT, same image timeline: reuse saved MOT predictions")
    score.add_argument("--manifest", required=True)
    score.add_argument("--original-manifest", required=True)
    score.add_argument("--predictions", required=True, help="Directory with sequence.txt MOT predictions")
    score.add_argument("--output", required=True)
    score.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if hasattr(args, "workers") and args.workers < 0:
        p.error("workers must be nonnegative")
    if args.command not in ("cache", "prepare", "import-cache", "verify-cache") and args.workers < 1:
        p.error("CPU replay requires at least one worker")
    return args


def dispatch(args):
    if args.command == "prepare":
        from .dataset import prepare
        prepare(args.data, args.split, args.role, args.output, args.hash_images)
    elif args.command == "cache":
        if min(args.batch_size, args.block_frames, args.height, args.width) < 1:
            raise ValueError("batch-size, block-frames, height and width must be positive")
        from .cache import yolox_cache
        yolox_cache(args)
    elif args.command == "import-cache":
        if args.format == "val-mm" and not args.detector_sha256:
            raise ValueError("val-mm requires --detector-sha256 from the original cache provenance")
        from .cache import import_legacy
        import_legacy(args)
    elif args.command == "verify-cache":
        from .cache import verify
        from .io import read
        verify(args.cache, read(args.manifest))
        print("Cache verified")
    elif args.command == "validate":
        from .io import read
        from .runner import freeze, replay
        manifest, config = read(args.manifest), read(args.config)
        if manifest["role"] != "val":
            raise ValueError("validate only accepts a val manifest")
        result = replay(manifest, args.cache, config, args.output, args.workers)
        if result:
            freeze(config, manifest, args.cache, result["summary"], Path(args.output)/"selection.json", "published config revalidated")
            print(result["summary"])
    elif args.command == "tune":
        from .tuning import tune
        tune(args)
    elif args.command == "ablate":
        from .tuning import ablate
        ablate(args)
    elif args.command == "evaluate":
        from .runner import evaluate_test
        evaluate_test(args)
    elif args.command == "rescore":
        from .runner import rescore
        rescore(args)


if __name__ == "__main__":
    main()
