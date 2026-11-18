"""Small synthetic regression tests; no dataset or detector weights required."""
import contextlib
import copy
import io
import json
import os
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
from exsort.runner import replay, replay_fps, track_one, worker_manifest
from exsort.scoring import evaluate_sequence, summary

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(root):
    data = root/"data"
    for split, name, n in [("train", "train_scene", 24), ("test", "test_scene", 12)]:
        folder = data/split/name
        (folder/"images").mkdir(parents=True)
        (folder/"mot/gt").mkdir(parents=True)
        if split == "train":
            save(folder/"meta/DONE.json", {"output_integer_fps": 22})
        else:
            (folder/"seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")
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
        self.config = read(ROOT/"configs/final.json")

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
        save(search, dict(numeric_search=dict(IOU=[.25, .35]), trials=3, seed=7))
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
        modified["config"]["params"]["IOU"] = .5
        save(tune/"selection.json", modified)
        command(*args, success=False)

    def test_config_keys_and_fps_are_checked(self):
        with self.assertRaisesRegex(ValueError, "positive sequence FPS"):
            create_tracker(self.config)
        for change in (dict(LOW=.1), dict(DET=.7), dict(MIN_HITS=0)):
            config = copy.deepcopy(self.config)
            config["params"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                create_tracker(config, fps=25)
        self.assertEqual(self.val["sequences"]["train_scene"]["fps"], 22)
        self.assertEqual(self.test["sequences"]["test_scene"]["fps"], 25)
        old_manifest = copy.deepcopy(self.val)
        del old_manifest["sequences"]["train_scene"]["fps"]
        self.assertEqual(replay_fps(old_manifest, "train_scene"), 22)
        self.assertEqual(image_identity(self.val), image_identity(old_manifest))

    def test_unique_and_ambiguous_recovery(self):
        empty = np.zeros((512, 512), bool)
        det = lambda *xs: [dict(box=np.array([x, .5, .1, .4], np.float32), conf=.9) for x in xs]
        # A new track confirmed next to one lost track takes its ID.
        model = create_tracker(self.config, fps=22)
        for k, xs in enumerate([(.30,), (.30,), (.30,), (.30, .45), (.45,), (.45,)], 1):
            out = model.step(det(*xs), 512, 512, empty)
        self.assertEqual([(t.id, t.kf.time_since_update) for t in out], [(1, 0)])
        self.assertEqual(len(model.tracks), 1)
        # A lost track blocks a new track at a nearby position.
        model.step([], 512, 512, empty)
        model.step(det(.52), 512, 512, empty)
        self.assertEqual([t.id for t in model.tracks], [1])
        # With two lost candidates the new track keeps its own ID.
        model = create_tracker(self.config, fps=22)
        for xs in [(.2, .6), (.2, .6), (.2, .6), (.2, .6, .4), (.4,), (.4,)]:
            out = model.step(det(*xs), 512, 512, empty)
        self.assertEqual([(t.id, t.kf.time_since_update) for t in out], [(3, 0)])
        self.assertEqual(sorted(t.id for t in model.tracks), [1, 2, 3])

    def test_glare_output_limit_and_memory(self):
        empty = np.zeros((512, 512), bool)
        glare = empty.copy()
        glare[160, 256] = True  # inside the head band of the box below
        rows = [dict(box=[.5, .5, .2, .4], conf=.9)]
        limit = self.config["params"]["GLARE_COAST_SECONDS"]
        for fps in (19, 22, 25):
            with self.subTest(fps=fps):
                model = create_tracker(self.config, fps=fps)
                for _ in range(3):
                    model.step(rows, 512, 512, empty)
                # One clock from the last match; a box is reported only in glare frames.
                for age in range(1, int(2 * limit * fps) + 2):
                    has_glare = age % 3 != 0
                    out = model.step([], 512, 512, glare if has_glare else empty)
                    self.assertEqual([t.id for t in out], [1] if has_glare and age <= limit * fps else [])
                    self.assertEqual([t.id for t in model.tracks], [1])
                out = model.step(rows, 512, 512, empty)
                self.assertEqual([t.id for t in out], [1])

    def test_credibility_rules(self):
        empty = np.zeros((512, 512), bool)
        glare = empty.copy()
        glare[160, 256] = True
        det = lambda conf: [dict(box=[.5, .5, .2, .4], conf=conf)]
        fps = 20
        # Confirmation counts only credible detections (score >= INIT): 0.9, 0.4, 0.4 is not enough.
        model = create_tracker(self.config, fps=fps)
        outs = [model.step(det(c), 512, 512, empty) for c in (.9, .4, .4, .9, .9)]
        self.assertEqual([len(o) for o in outs], [0, 0, 0, 0, 1])
        # A detection below INIT cannot start a track.
        model = create_tracker(self.config, fps=fps)
        model.step(det(.5), 512, 512, empty)
        self.assertEqual(model.tracks, [])
        # A glare box needs a credible mean score of the matched detections; there is no output without glare.
        for scores, coasts in [((.9, .9, .9), True), ((.9, .7, .7) + (.3,) * 6, False)]:
            with self.subTest(scores=scores):
                model = create_tracker(self.config, fps=fps)
                for c in scores:
                    model.step(det(c), 512, 512, empty)
                out = model.step([], 512, 512, glare)
                self.assertEqual([t.id for t in out], [1] if coasts else [])
                self.assertEqual(model.step([], 512, 512, empty), [])
        # Low-score matches keep the track alive, but its box stops after MAX_AGE frames without a credible match.
        model = create_tracker(self.config, fps=fps)
        for _ in range(3):
            model.step(det(.9), 512, 512, empty)
        max_age = self.config["params"]["MAX_AGE"]
        for k in range(1, max_age + 2):
            out = model.step(det(.3), 512, 512, empty)
            self.assertEqual([t.id for t in out], [1] if k <= max_age else [])
        self.assertEqual([t.id for t in model.tracks], [1])
        out = model.step(det(.9), 512, 512, empty)
        self.assertEqual([t.id for t in out], [1])

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
