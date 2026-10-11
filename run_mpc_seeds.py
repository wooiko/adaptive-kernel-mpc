"""Part 2, 20 seeds: every configuration of run_mpc.PLAN on S1-S6 with the frozen settings of S0.

    python run_mpc_seeds.py                    # seeds 0-19, writes outputs/mpc_seeds.json
    python run_mpc_seeds.py --cache DIR        # also keep each finished run in DIR and skip runs found there

Two worker processes, one BLAS thread each (set before numpy is imported). The oracle NMPC on the true
model runs only on the reference seed (run_mpc.py, outputs/mpc_metrics.json), not here (TZ 7).
Seed 0 here repeats the runs of run_mpc.py for seed 0; the two files must agree on every non-time field
(tests/test_mpc.py). Time fields (step_ms, ident_ms, wall_s, wall_s_total) change between runs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import mpc  # noqa: E402
import run_mpc as rm  # noqa: E402
import safety  # noqa: E402

N_SEEDS = 20
OUT_FILE = rm.OUT / "mpc_seeds.json"


def commission(seed):
    with threadpool_limits(1):
        c, h = rm.commissioning(seed)
    return seed, c, h


def cache_name(seed, scen, name):
    return f"s{seed:02d}_{scen}_{name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    lim = safety.tc_max()
    s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(rm.WORKERS) as ex:
        comm = {seed: (c, h) for seed, c, h in ex.map(commission, seeds)}
    t_comm = time.perf_counter() - t0
    sizes = {k: len(f(0, lim["tc_max"])["r"]) for k, f in cl.SCENARIOS.items()}
    jobs, results, logs = [], {}, {}   # logs: S4 only, for the inputs moved by spikes (run_mpc.s4_moves)
    for seed in seeds:
        for scen, names in rm.PLAN.items():
            for name in names:
                f = args.cache / cache_name(seed, scen, name) if args.cache else None
                if f is not None and f.exists():
                    rec = json.loads(f.read_text())
                    results.setdefault(seed, {}).setdefault(scen, {})[name] = rec["m"]
                    if rec["log"] is not None:
                        logs.setdefault(seed, {})[(scen, name)] = rec["log"]
                else:
                    jobs.append((comm[seed][0], seed, lim["tc_max"], scen, name, s, mpc.NP, scen == "S4"))
    jobs.sort(key=lambda j: (-sizes[j[3]] * (4 if j[4].startswith("NMPC") else 1), j[1]))   # longest first
    print(f"commissioning {t_comm:.0f} s; {len(jobs)} runs to do", flush=True)
    done = 0
    with ProcessPoolExecutor(rm.WORKERS) as ex:
        futs = {ex.submit(rm.job, j): j for j in jobs}
        for fu in as_completed(futs):
            seed = futs[fu][1]
            scen, name, m, lg = fu.result()
            results.setdefault(seed, {}).setdefault(scen, {})[name] = m
            if lg is not None:
                logs.setdefault(seed, {})[(scen, name)] = lg
            if args.cache:
                (args.cache / cache_name(seed, scen, name)).write_text(json.dumps({"m": m, "log": lg}))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} runs, {time.perf_counter() - t0:.0f} s", flush=True)
    wall = time.perf_counter() - t0
    out = {}
    for seed in seeds:
        sk = results[seed]
        rm.s4_moves(sk, logs.get(seed, {}))
        out[str(seed)] = {"startup_block_sha256_16": comm[seed][1],
                          "commissioning": {"sigma": comm[seed][0]["sigma"], "lam": comm[seed][0]["lam"]},
                          "results": {scen: {name: sk[scen][name] for name in rm.PLAN[scen]} for scen in sorted(sk)}}
    res = {"seeds": seeds, "machine": rm.machine(), "settings": rm.settings_block(lim, s),
           "tc_max": {k: lim[k] for k in ("tc_max", "worst_limit", "worst_vertex", "margin", "ca_nominal_at_tc_max")},
           "wall_s_total": wall, "by_seed": out}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"done in {wall:.0f} s", flush=True)


if __name__ == "__main__":
    main()
