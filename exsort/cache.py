"""Content-addressed detector + measured saturation-mask cache."""
import os
from pathlib import Path
import cv2
import numpy as np
from .dataset import assert_images
from .io import key, read, save, sha

MASK = dict(width=512, height=512, V_min=235, S_max=89, resize="INTER_AREA-before-HSV")


def image_identity(manifest):
    return key(dict(role=manifest["role"], split=manifest["split_signature"],
                    sequences={s: {k: v for k, v in q.items() if k != "gt_sha256"}
                               for s, q in manifest["sequences"].items()}))


def open_cache(folder, manifest):
    folder = Path(folder)
    meta = read(folder / "cache.json")
    if meta["image_identity"] != image_identity(manifest):
        raise ValueError("Cache image/timeline signature differs from manifest")
    if meta["mask"] != MASK:
        raise ValueError("Saturation mask definition differs")
    return meta


def ledger(folder, name, identity):
    """Recover even an atomically committed block not yet added to index.json."""
    root = Path(folder) / name
    blocks, cursor = [], 0
    for path in sorted(root.glob("[0-9]*.npz")):
        with np.load(path, allow_pickle=False) as z:
            start, end = int(z["start"]), int(z["end"])
            off, det, masks = z["offsets"], z["detections"], z["masks"]
            if start != cursor or end <= start or str(z["identity"]) != identity:
                raise ValueError(f"Non-contiguous/incompatible cache block: {path}")
            if len(off) != end-start+1 or off[0] != 0 or off[-1] != len(det) or (np.diff(off) < 0).any():
                raise ValueError(f"Invalid offsets: {path}")
            if det.ndim != 2 or det.shape[1] != 5 or masks.shape != (end-start, 32768):
                raise ValueError(f"Invalid cache array shapes: {path}")
            if not np.isfinite(det).all() or (det[:, 2:4] <= det[:, :2]).any():
                raise ValueError(f"Invalid detector boxes: {path}")
            if ((det[:, 4] < 0) | (det[:, 4] > 1)).any() or masks.dtype != np.uint8:
                raise ValueError(f"Invalid scores/masks: {path}")
        blocks.append(dict(file=path.name, start=start, end=end, sha256=sha(path)))
        cursor = end
    save(root / "index.json", dict(identity=identity, frames=cursor, blocks=blocks))
    return blocks


def write_block(root, start, end, identity, offsets, detections, masks):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{start:08d}.tmp"
    with tmp.open("wb") as f:
        np.savez(f, start=start, end=end, identity=identity, offsets=np.asarray(offsets, np.int64),
                 detections=np.asarray(detections, np.float32).reshape(-1, 5), masks=np.asarray(masks, np.uint8))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, root / f"{start:08d}.npz")


def verify(folder, manifest, check_images=True):
    meta = open_cache(folder, manifest)
    for s, q in manifest["sequences"].items():
        if check_images:
            assert_images(manifest, s)
        blocks = ledger(folder, s, meta["identity"])
        if not blocks or blocks[-1]["end"] != q["frames"]:
            raise ValueError(f"Incomplete cache: {s}")
    save(Path(folder) / "verified.json", dict(image_identity=meta["image_identity"],
         identity=meta["identity"], sequences=len(manifest["sequences"]), frames=manifest["frames"]))
    return meta


def saturation(image):
    hsv = cv2.cvtColor(cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    return np.packbits(((hsv[:, :, 2] >= 235) & (hsv[:, :, 1] <= 89)).reshape(-1))


def preprocess(image, size, dtype=np.float16):
    """ByteTrack/TrackTrack RGB ImageNet normalization, single resize, top-left padding."""
    h, w = image.shape[:2]
    ratio = min(size[0] / h, size[1] / w)
    resized = cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((*size, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    lut = (((np.arange(256, dtype=np.float64)[:, None] / 255.) - np.array([.485, .456, .406])) /
           np.array([.229, .224, .225])).astype(np.float32).astype(dtype)
    return np.ascontiguousarray(lut[padded[:, :, ::-1], np.arange(3)].transpose(2, 0, 1))


class FrameBatches:
    def __init__(self, manifest, name, start, batch, size, fp16):
        self.manifest, self.name, self.start = manifest, name, start
        self.batch, self.size, self.fp16 = batch, size, fp16

    def __len__(self):
        return (self.manifest["sequences"][self.name]["frames"] - self.start + self.batch-1) // self.batch

    def __getitem__(self, i):
        import torch
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        q = self.manifest["sequences"][self.name]
        root = Path(self.manifest["data_root"]) / q["split"] / self.name / "images"
        start = self.start + i*self.batch
        end = min(start+self.batch, q["frames"])
        inputs, masks = [], []
        for k in range(start, end):
            image = cv2.imread(str(root / q["images"][k]))
            if image is None or image.shape[:2] != (q["height"], q["width"]):
                raise ValueError(f"Unreadable image or dimensions changed: {self.name}/{k}")
            masks.append(saturation(image))
            inputs.append(preprocess(image, self.size, np.float16 if self.fp16 else np.float32))
        return dict(start=start, end=end, images=torch.from_numpy(np.stack(inputs)), masks=np.asarray(masks))


def yolox_cache(args):
    import importlib.util
    import sys
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    manifest = read(args.manifest)
    out = Path(args.output)
    source = Path(args.yolox_root).resolve()
    if not (source / "yolox").is_dir():
        raise ValueError("--yolox-root must contain the TrackTrack/ByteTrack yolox package")
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("exsort_detector_exp", args.exp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(20261028)
    torch.backends.cudnn.benchmark = True
    exp = module.Exp()
    if exp.num_classes != 1:
        raise ValueError("This adapter requires a person-only detector")
    model = exp.get_model()
    # Checkpoints are executable pickle data; only use weights you trust.
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
    del ck
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install a CUDA-compatible PyTorch build")
    if args.fp16 and not args.device.startswith("cuda"):
        raise ValueError("Use --no-fp16 for CPU inference")
    model = model.to(args.device).eval()
    if args.fp16:
        model.half()
    size = (args.height, args.width)
    meta = dict(schema=1, image_identity=image_identity(manifest),
                detector_sha256=sha(args.weights), exp_sha256=sha(args.exp),
                detector_source_sha256=key({p.relative_to(source).as_posix(): sha(p) for p in sorted((source / "yolox").rglob("*.py"))}),
                adapter_sha256=sha(__file__), size=size, fp16=args.fp16,
                score_floor=.001, nms=.65, cap=100, mask=MASK,
                preprocessing="single-resize/top-left114/BGR-to-RGB/ImageNet-mean-std; no Conv-BN fusion",
                torch=torch.__version__, torchvision=torchvision.__version__,
                device=args.device, gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu")
    meta = read_roundtrip(meta)
    meta["identity"] = key(meta)
    if (out / "cache.json").exists() and read(out / "cache.json") != meta:
        raise ValueError("Cache protocol changed; use a new output directory")
    save(out / "cache.json", meta)
    with torch.inference_mode():
        for name, q in sorted(manifest["sequences"].items(), key=lambda x: -x[1]["frames"]):
            assert_images(manifest, name)
            committed = ledger(out, name, meta["identity"])
            cursor = committed[-1]["end"] if committed else 0
            batch = args.batch_size
            while cursor < q["frames"]:
                if (out / "STOP").exists():
                    print("Paused at a committed cache block. Remove STOP and repeat the command.")
                    return
                dataset = FrameBatches(manifest, name, cursor, batch, size, args.fp16)
                options = dict(batch_size=None, num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
                if args.workers:
                    options.update(prefetch_factor=2)
                loader = DataLoader(dataset, **options)
                start, offsets, dets, masks = cursor, [0], [], []
                try:
                    for item in loader:
                        pred = model(item["images"].to(args.device, non_blocking=True))
                        xy = torch.empty_like(pred[:, :, :4])
                        xy[:, :, :2] = pred[:, :, :2] - pred[:, :, 2:4]/2
                        xy[:, :, 2:] = pred[:, :, :2] + pred[:, :, 2:4]/2
                        conf = pred[:, :, 4] * pred[:, :, 5]
                        ratio = min(size[0]/q["height"], size[1]/q["width"])
                        for i in range(int(item["end"])-int(item["start"])):
                            keep = conf[i] >= .001
                            boxes = xy[i, keep]
                            scores = pred[i, keep, 4].float() * pred[i, keep, 5].float()
                            ids = torchvision.ops.nms(boxes.float(), scores, .65)[:100]
                            arr = torch.cat((boxes[ids].float()/ratio, scores[ids, None]), 1).cpu().numpy()
                            dets.append(arr)
                            offsets.append(offsets[-1] + len(arr))
                        masks.extend(item["masks"].numpy())
                        end = int(item["end"])
                        if end-start >= args.block_frames or end == q["frames"]:
                            write_block(out/name, start, end, meta["identity"], offsets, np.concatenate(dets), masks)
                            cursor = end
                            save(out / "progress.json", dict(sequence=name, frame=cursor, frames=q["frames"], batch_size=batch))
                            print(f"Cached {name}: {cursor}/{q['frames']}", flush=True)
                            start, offsets, dets, masks = cursor, [0], [], []
                            if (out / "STOP").exists():
                                return
                        del pred, xy, conf
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch == 1:
                        raise
                    # Discard only uncommitted frames and retry from last block.
                    del loader
                    item = None
                    import gc
                    pred = xy = conf = boxes = scores = ids = arr = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch = max(1, batch//2)
                    print(f"OOM: retry {name} from committed frame {cursor}, batch={batch}", flush=True)
            ledger(out, name, meta["identity"])
    verify(out, manifest)


def read_roundtrip(value):
    from .io import native
    import json
    return json.loads(json.dumps(native(value)))


def import_legacy(args):
    """Read only trusted local legacy caches; never infer or synthesize a mask."""
    if not args.trust_legacy_images:
        raise ValueError("Legacy caches lack full image hashes. Verify their provenance, then pass --trust-legacy-images.")
    manifest = read(args.manifest)
    source, out = Path(args.source), Path(args.output)
    provenance = dict(format=args.format, trusted_legacy_images=True,
                      limitation="Legacy source has no complete image-content signature; caller attests correspondence.")
    mapping = read(args.mask_map).get("mapping", {}) if args.mask_map else {}
    if args.format == "val-mm":
        if manifest["role"] != "val":
            raise ValueError("val-mm input only supports validation")
        old = read(source / "meta.json")
        provenance["source_meta_sha256"] = sha(source / "meta.json")
        provenance["detector_sha256"] = args.detector_sha256
        provenance["detector_note"] = "User-supplied legacy weight identity; not inferred from pickle"
    else:
        old = read(source)
        if manifest["role"] != "test":
            raise ValueError("test-blocks input only supports full test sequences")
        provenance.update(source_protocol_sha256=sha(source), detector_sha256=old.get("detector_model_sha256", old.get("model_sha256")))
    meta = dict(schema=1, image_identity=image_identity(manifest), score_floor=.001, nms=.65, cap=100,
                mask=MASK, provenance=provenance)
    meta["identity"] = key(meta)
    if (out/"cache.json").exists() and read(out/"cache.json") != meta:
        raise ValueError("Import protocol changed; choose another output")
    save(out/"cache.json", meta)
    for s, q in manifest["sequences"].items():
        assert_images(manifest, s)
        existing = ledger(out, s, meta["identity"])
        cursor = existing[-1]["end"] if existing else 0
        if cursor == q["frames"]:
            continue
        if args.format == "val-mm":
            import pickle
            info = old["sequences"][s]
            if info["half"] != q["start"] or info["nval"] != q["frames"] or info["n"] != q["n_total"]:
                raise ValueError(f"Timeline mismatch: {s}")
            with (source/"mm"/(s+".pkl")).open("rb") as f:
                T, W, H, off, det, _gt, _features = pickle.load(f)
            if (T, W, H) != (q["frames"], q["width"], q["height"]):
                raise ValueError(f"Dimension mismatch: {s}")
            masks = np.load(source/"mm"/(s+".npy"), mmap_mode="r")
            for start in range(cursor, T, 512):
                end = min(start+512, T)
                write_block(out/s, start, end, meta["identity"], off[start:end+1]-off[start], det[off[start]:off[end], :5], masks[start:end])
        else:
            info = old["sequences"][s]
            if (info["frames"], info["width"], info["height"]) != (q["frames"], q["width"], q["height"]):
                raise ValueError(f"Timeline/dimension mismatch: {s}")
            for b in info["blocks"]:
                if b["end"] <= cursor:
                    continue
                path = Path(b["path"])
                if sha(path) != b["sha256"]:
                    raise ValueError(f"Legacy detection hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as z:
                    off, det = z["off65"], z["det65"][:, :5]
                    if int(z["start"]) != b["start"] or int(z["end"]) != b["end"]:
                        raise ValueError("Legacy block timeline differs from protocol")
                    if info["rebuild_image_cache"]:
                        masks = z["masks"]
                    else:
                        pieces = []
                        for c in info["image_chunks"]:
                            a, e = max(c["start"], b["start"]), min(c["end"], b["end"])
                            if a >= e:
                                continue
                            cp = Path(c["path"])
                            if not cp.exists():
                                cp = Path(mapping.get(c["path"], c["path"]))
                            if sha(cp) != c["sha256"]:
                                raise ValueError(f"Legacy mask hash mismatch: {cp}")
                            with np.load(cp, allow_pickle=False) as cz:
                                pieces.append(cz["masks"][a-c["start"]:e-c["start"]])
                        masks = np.concatenate(pieces)
                    write_block(out/s, b["start"], b["end"], meta["identity"], off, det, masks)
        print(f"Imported {s}", flush=True)
    verify(out, manifest)
