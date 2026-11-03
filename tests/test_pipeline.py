"""Small synthetic regression tests; no dataset or detector weights required."""
import contextlib
import copy
import io
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import cv2
import numpy as np
from exsort.api import create_tracker
from exsort.cache import MASK, image_identity, ledger, saturation, write_block
from exsort.dataset import current_gt, prepare
from exsort.io import key, read, save, sha
from exsort.runner import replay, track_one, worker_manifest
from exsort.scoring import evaluate_sequence, summary

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(root):
    data = root/"data"
    for split, name, n in [("train", "train_scene", 24), ("test", "test_scene", 12)]:
        folder = data/split/name
        (folder/"images").mkdir(parents=True)
        (folder/"mot/gt").mkdir(parents=True)
        rows = []
        for k in range(n):
            image = np.full((80, 100, 3), 20, dtype=np.uint8)
            image[20:23, 40:45] = 255
            cv2.imwrite(str(folder/"images"/f"frame_{k:06d}.jpg"), image)
            rows.append(f"{k+1},1,30,20,20,40,1,1,1\n")
        (folder/"mot/gt/gt.txt").write_text("".join(rows))
    split = root/"split.json"
    save(split, dict(train=["train_scene"], test=["test_scene"]))
    with contextlib.redirect_stdout(io.StringIO()):
        val = prepare(data, split, "val", root/"val.json")
        test = prepare(data, split, "test", root/"test.json")
    for role, m in [("val", val), ("test", test)]:
        cache = root/(role+"_cache")
        meta = dict(schema=1, image_identity=image_identity(m), score_floor=.001, nms=.65, cap=100,
                    mask=MASK, detector_sha256="a"*64, fixture=True)
        meta["identity"] = key(meta)
        save(cache/"cache.json", meta)
        for name, q in m["sequences"].items():
            for start in range(0, q["frames"], 4):
                end = min(start+4, q["frames"])
                rows = np.tile([30, 20, 50, 60, .9], (end-start, 1))
                masks = []
                for k in range(start, end):
                    image = cv2.imread(str(data/q["split"]/name/"images"/q["images"][k]))
                    masks.append(saturation(image))
                write_block(cache/name, start, end, meta["identity"], np.arange(end-start+1), rows, masks)
            ledger(cache, name, meta["identity"])
    return val, test


def command(*args, success=True):
    result = subprocess.run([sys.executable, "-m", "exsort", *map(str, args)], cwd=ROOT,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"), capture_output=True, text=True, encoding="utf8")
    if success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not success and not result.returncode:
        raise AssertionError("Expected command to reject invalid protocol")
    return result


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.val, self.test = make_fixture(self.root)
        self.config = read(ROOT/"configs/selected.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_glare_is_measured_and_gt_changes_are_rejected(self):
        q = self.val["sequences"]["train_scene"]
        self.assertEqual((q["start"], q["frames"]), (12, 12))
        self.assertEqual(current_gt(self.val, "train_scene")[0, 0], 1)
        gt = self.root/"data/train/train_scene/mot/gt/gt.txt"
        gt.write_text(gt.read_text().replace(",30,20,", ",31,20,"))
        with self.assertRaisesRegex(ValueError, "GT changed"):
            current_gt(self.val, "train_scene")
        self.assertGreater(np.unpackbits(saturation(np.full((20, 20, 3), 255, np.uint8))).sum(), 0)
        self.assertEqual(np.unpackbits(saturation(np.zeros((20, 20, 3), np.uint8))).sum(), 0)

    def test_checkpoint_resume_matches_uninterrupted_predictions(self):
        cache = self.root/"val_cache"
        full = self.root/"full"
        result = replay(self.val, cache, self.config, full, workers=1)
        protocol = read(full/"protocol.json")
        interrupted = self.root/"interrupted"
        interrupted.mkdir()
        (interrupted/"STOP").touch()
        job = (worker_manifest(self.val, "train_scene"), str(cache), self.config, str(interrupted), "train_scene", protocol["track_identity"])
        partial = track_one(job)
        self.assertTrue(partial["paused"])
        self.assertEqual(partial["frames"], 4)
        # Simulate uncommitted output after a process interruption.
        path = interrupted/"work/train_scene/predictions.partial.txt"
        with path.open("ab") as f:
            f.write(b"uncommitted garbage")
        (interrupted/"STOP").unlink()
        track_one(job)
        self.assertEqual(sha(interrupted/"trackers/exsort/data/train_scene.txt"), sha(full/"trackers/exsort/data/train_scene.txt"))
        self.assertEqual(result["summary"]["IDSW"], 0)

    def test_cli_tune_freeze_test_resume_and_tamper_rejection(self):
        search = self.root/"search.json"
        save(search, dict(numeric_search=dict(DET=[.23, .28]), trials=3, seed=7))
        tune = self.root/"tuning"
        command("tune", "--manifest", self.root/"val.json", "--cache", self.root/"val_cache",
                "--search", search, "--output", tune, "--workers", 1)
        self.assertTrue(read(tune/"complete.json")["completed"])
        self.assertFalse(read(tune/"complete.json")["test_accessed"])
        selection_sha = sha(tune/"selection.json")
        command("tune", "--manifest", self.root/"val.json", "--cache", self.root/"val_cache",
                "--search", search, "--output", tune, "--workers", 1)
        self.assertEqual(sha(tune/"selection.json"), selection_sha)
        output = self.root/"test_eval"
        args = ["evaluate", "--manifest", self.root/"test.json", "--cache", self.root/"test_cache",
                "--selection", tune/"selection.json", "--output", output, "--workers", 1]
        command(*args)
        first_sha = sha(output/"trackers/exsort/data/test_scene.txt")
        command(*args)
        self.assertEqual(sha(output/"trackers/exsort/data/test_scene.txt"), first_sha)
        self.assertIn("Count", read(output/"all_trackeval_metrics.json")["combined"])
        command("tune", "--manifest", self.root/"test.json", "--cache", self.root/"test_cache",
                "--search", search, "--output", self.root/"invalid", "--workers", 1, success=False)
        modified = read(tune/"selection.json")
        modified["config"]["params"]["DET"] = .8
        save(tune/"selection.json", modified)
        command(*args, success=False)

    def test_unique_and_ambiguous_recovery(self):
        empty = np.zeros((512, 512), bool)
        config = copy.deepcopy(self.config)
        config["params"].update(BIRTH_POLICY="iom_active", INIT=0.)
        model = create_tracker(config)
        for k in range(1, 7):
            box = np.array([.30 if k <= 3 else .45, .5, .1, .4], np.float32)
            out = model.step([dict(box=box, conf=.9)], 512, 512, k, empty)
        self.assertEqual((len(model.tracks), len(out), out[0].id), (1, 1, 1))
        model = create_tracker(config)
        for k in range(1, 4):
            model.step([dict(box=np.array([x, .5, .1, .6], np.float32), conf=.9) for x in (.2, .6)], 512, 512, k, empty)
        for k in range(4, 7):
            out = model.step([dict(box=np.array([.4, .5, .1, .6], np.float32), conf=.9)], 512, 512, k, empty)
        self.assertEqual((len(out), out[0].id, len(model.tracks)), (1, 3, 3))

    def test_grouped_scoring_matches_native_trackeval(self):
        import trackeval
        gt = np.array([[1, 1, 10, 10, 20, 40, 1, 1, 1], [2, 1, 11, 10, 20, 40, 1, 1, 1],
                       [3, 1, 12, 10, 20, 40, 1, 1, 1]], float)
        pred = np.array([[1, 1, 10, 10, 20, 40], [2, 2, 11, 10, 20, 40], [3, 2, 12, 10, 20, 40],
                         [3, 3, 70, 10, 10, 10]], float)
        target = self.root/"native"
        (target/"gt/S/gt").mkdir(parents=True)
        (target/"trackers/exsort/data").mkdir(parents=True)
        np.savetxt(target/"gt/S/gt/gt.txt", gt, delimiter=",")
        (target/"gt/S/seqinfo.ini").write_text("[Sequence]\nname=S\nseqLength=3\nimWidth=100\nimHeight=80\nframeRate=25\n")
        rows = np.column_stack([pred, np.ones(4), -np.ones((4, 3))])
        np.savetxt(target/"trackers/exsort/data/S.txt", rows, delimiter=",")
        ec = trackeval.Evaluator.get_default_eval_config()
        ec.update(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False,
                  OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False)
        dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=str(target/"gt"), TRACKERS_FOLDER=str(target/"trackers"),
                  TRACKERS_TO_EVAL=["exsort"], CLASSES_TO_EVAL=["pedestrian"], BENCHMARK="MOT17", SPLIT_TO_EVAL="test",
                  SKIP_SPLIT_FOL=True, DO_PREPROC=False, PRINT_CONFIG=False, TRACKER_SUB_FOLDER="data", SEQ_INFO={"S": 3})
        with contextlib.redirect_stdout(io.StringIO()):
            native, status = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],
                [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
        self.assertEqual(status["MotChallenge2DBox"]["exsort"], "Success")
        expected = native["MotChallenge2DBox"]["exsort"]["COMBINED_SEQ"]["pedestrian"]
        actual = evaluate_sequence(gt, pred, 3)
        for family in expected:
            for field in expected[family]:
                np.testing.assert_allclose(actual[family][field], expected[family][field], atol=1e-12, rtol=1e-12, err_msg=f"{family}.{field}")

    def test_legacy_val_and_test_mask_mapping_import(self):
        from argparse import Namespace
        from exsort.cache import import_legacy, verify
        legacy = self.root/"legacy_val"
        (legacy/"mm").mkdir(parents=True)
        save(legacy/"meta.json", dict(sequences={"train_scene": dict(n=24, half=12, nval=12)}))
        det = np.tile(np.array([30, 20, 50, 60, .9], np.float32), (12, 1))
        masks = np.zeros((12, 32768), dtype=np.uint8)
        masks[:, 100] = 255
        with (legacy/"mm/train_scene.pkl").open("wb") as f:
            # Deliberately stale/invalid cached GT is ignored in favor of current manifest GT.
            pickle.dump((12, 100, 80, np.arange(13), det, "stale_gt_not_for_scoring", None), f)
        np.save(legacy/"mm/train_scene.npy", masks)
        args = Namespace(trust_legacy_images=True, manifest=self.root/"val.json", source=legacy,
                         output=self.root/"imported_val", format="val-mm", mask_map=None, detector_sha256="a"*64)
        with contextlib.redirect_stdout(io.StringIO()):
            import_legacy(args)
        verify(args.output, self.val)
        with np.load(args.output/"train_scene/00000000.npz") as z:
            np.testing.assert_array_equal(z["detections"], det)
            np.testing.assert_array_equal(z["masks"], masks)
        block = self.root/"old_detections.npz"
        np.savez(block, start=0, end=12, off65=np.arange(13), det65=det)
        actual_masks = self.root/"moved_masks.npz"
        np.savez(actual_masks, masks=masks)
        vanished = str(self.root/"vanished_masks.npz")
        protocol = self.root/"old_test_protocol.json"
        save(protocol, dict(detector_model_sha256="a"*64, sequences={"test_scene": dict(frames=12, width=100, height=80,
              rebuild_image_cache=False, blocks=[dict(path=str(block), sha256=sha(block), start=0, end=12)],
              image_chunks=[dict(path=vanished, sha256=sha(actual_masks), start=0, end=12)])}))
        mapping = self.root/"mask_map.json"
        save(mapping, dict(mapping={vanished: str(actual_masks)}))
        args = Namespace(trust_legacy_images=True, manifest=self.root/"test.json", source=protocol,
                         output=self.root/"imported_test", format="test-blocks", mask_map=mapping, detector_sha256=None)
        with contextlib.redirect_stdout(io.StringIO()):
            import_legacy(args)
        verify(args.output, self.test)
        with np.load(args.output/"test_scene/00000000.npz") as z:
            np.testing.assert_array_equal(z["masks"], masks)
        # A removed mask cannot be silently replaced with zeros.
        args.output = self.root/"missing_mask_import"
        args.mask_map = None
        with self.assertRaises(FileNotFoundError):
            import_legacy(args)

    def test_new_gt_can_rescore_without_replay(self):
        output = self.root/"original_test"
        replay(self.test, self.root/"test_cache", self.config, output, workers=1)
        original = read(output/"metrics.json")["summary"]
        gt = self.root/"data/test/test_scene/mot/gt/gt.txt"
        gt.write_text(gt.read_text().replace(",30,20,", ",31,20,"))
        command("prepare", "--data", self.root/"data", "--split", self.root/"split.json", "--role", "test", "--output", self.root/"new_test.json")
        command("rescore", "--manifest", self.root/"new_test.json", "--original-manifest", self.root/"test.json",
                "--predictions", output/"trackers/exsort/data", "--output", self.root/"rescored", "--workers", 1)
        updated = read(self.root/"rescored/metrics.json")["summary"]
        self.assertLess(updated["HOTA"], original["HOTA"])
        self.assertFalse((self.root/"rescored/work").exists())


if __name__ == "__main__":
    unittest.main()
