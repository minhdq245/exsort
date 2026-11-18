"""Sequential SMAC3 search on validation only. Selection: highest validation MOTA."""
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
from .api import INTEGER, PARAMS, validate
from .cache import verify
from .io import key, read, save, source_signature
from .runner import freeze, replay


def ranking(row):
    """Highest MOTA first; ties go to higher HOTA, then higher IDF1, then fewer ID switches."""
    s = row["summary"]
    return (-s["MOTA"], -s["HOTA"], -s["IDF1"], s["IDSW"])


def tune(args):
    from ConfigSpace import ConfigurationSpace, Configuration, Float, Integer
    from smac import Scenario, HyperparameterOptimizationFacade as SMAC
    from smac.runhistory.dataclasses import TrialInfo, TrialValue, TrialKey
    from smac.runhistory.enumerations import StatusType
    manifest = read(args.manifest)
    if manifest["role"] != "val":
        raise ValueError("Tuning reads validation manifests only")
    initial = validate(read(args.initial_config))
    space = read(args.search)
    meta = verify(args.cache, manifest)
    root = Path(args.output)
    protocol = dict(manifest_signature=manifest["signature"], cache_identity=meta["identity"],
                    source_signature=source_signature(), initial=initial, search=space,
                    selection="max pooled validation MOTA; ties: HOTA, IDF1, fewer IDSW",
                    optimizer="SMAC3 2.4.1 RF + LogEI; sequential ask/tell, full validation every trial",
                    test_accessed=False)
    if (root/"protocol.json").exists() and read(root/"protocol.json") != protocol:
        raise ValueError("Tune protocol changed; choose another output directory")
    save(root/"protocol.json", protocol)
    if (root/"selection.json").exists():
        print("Completed; frozen selection:", root/"selection.json")
        return
    cs = ConfigurationSpace(seed=space["seed"])
    for name, limits in space["numeric_search"].items():
        if name not in PARAMS:
            raise ValueError(f"Unknown parameter {name}")
        cls = Integer if name in INTEGER else Float
        cs.add(cls(name, tuple(limits), default=initial["params"][name]))
    scenario = Scenario(cs, name="exsort_rf", output_directory=root/"smac", deterministic=True,
                        n_trials=space["trials"], seed=space["seed"])
    seed = Configuration(cs, values={k: initial["params"][k] for k in space["numeric_search"]})
    opt = SMAC(scenario, target_function=None,
               initial_design=SMAC.get_initial_design(scenario, n_configs=min(8, max(1, space["trials"]//4)), additional_configs=[seed]),
               intensifier=SMAC.get_intensifier(scenario, max_config_calls=1), overwrite=False, logging_level=30)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        def evaluate(config):
            # A finished trial is reused: replay checks its protocol, GT and prediction hashes.
            folder = root/"evaluations"/key(config)[:20]
            result = replay(manifest, args.cache, config, folder, pool=pool, validated=True)
            if result is None:
                raise RuntimeError("Replay paused. Remove STOP in trial directory and rerun.")
            return result

        evaluate(initial)
        pending = root/"pending.json"
        while opt.runhistory.finished < space["trials"]:
            if (root/"STOP").exists():
                print("Paused between trials. Remove STOP and repeat the same command.")
                return
            info = None
            if pending.exists() and read(pending):
                old = read(pending)
                candidate = TrialInfo(config=Configuration(cs, values=old["config"]), instance=old["instance"], seed=old["seed"], budget=old["budget"])
                cid = opt.runhistory.config_ids.get(candidate.config)
                try:
                    finished = opt.runhistory[TrialKey(cid, candidate.instance, candidate.seed, candidate.budget)].status == StatusType.SUCCESS
                except KeyError:
                    finished = False
                if not finished:
                    info = candidate
            if info is None:
                info = opt.ask()
            save(pending, dict(config=dict(info.config), instance=info.instance, seed=info.seed, budget=info.budget))
            config = copy.deepcopy(initial)
            for k, v in dict(info.config).items():
                config["params"][k] = int(v) if k in INTEGER else float(v)
            began = time.perf_counter()
            result = evaluate(config)
            s = result["summary"]
            cost = 100-s["MOTA"]
            opt.tell(info, TrialValue(cost=cost, time=time.perf_counter()-began,
                                     additional_info=dict(config_key=key(config), summary=s)))
            # SMAC persists runhistory on tell; pending file repairs interruption between ask/tell.
            save(pending, {})
            rows = [read(p) for p in (root/"evaluations").glob("*/metrics.json")]
            rows.sort(key=ranking)
            save(root/"ranked_results.json", rows)
            save(root/"best.json", rows[0])
            save(root/"state.json", dict(finished=opt.runhistory.finished, trials=space["trials"], best=rows[0]["summary"]))
            print(f"Trial {opt.runhistory.finished}/{space['trials']}: {s}", flush=True)
        rows = [read(p) for p in (root/"evaluations").glob("*/metrics.json")]
        rows.sort(key=ranking)
        best = rows[0]
        freeze(best["config"], manifest, args.cache, best["summary"], root/"selection.json")
        save(root/"complete.json", dict(completed=True, test_accessed=False, best=best["summary"]))
        print("Frozen selection:", root/"selection.json", best["summary"], flush=True)
