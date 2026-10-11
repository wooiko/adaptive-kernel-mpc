"""One-time validation of D14 (excitation window, TZ 14) on seeds never used before, with criteria fixed before the run.

The procedure and the criteria come from the independent analysis of the S2 steady-state offset (TZ 14, rule P14)
and are written here before the first validation run; after that run neither the code nor the parameters change
(a change means a new validation on seeds 60-79).

Configurations: F = "K-MPC adaptive, excitation window" (D14), "K-MPC static", "K-MPC adaptive".
Seeds 20-39; reserve 40-59 for one repetition.
Sets:
    V1  S2 unchanged                                      (closed_loop.scenario_s2)
    V2  D1: long time without excitation, random set-point (closed_loop.scenario_d1)
    V3  Ta: late Tf step after 1050 nominal samples        (closed_loop.scenario_ta)
    V4  S2r: S2 with the events in reverse order           (closed_loop.scenario_s2r)
    V5  S1, S3, S4, S6 of the repository
Criteria (steady-state error = closed_loop.segment_metrics, in 1e-3 mol/L; segments with a constant disturbance):
    a. |error| <= 1.0 on every such segment of every seed of V1-V4, configuration F;
    b. for every segment of V1-V4: |mean over seeds of (F - K-MPC static)| <= 0.15, and every single |F - static| <= 0.5;
    c. gamma = K_model / K_true >= 0.2 at every probe (every 10 samples) of every run of F in V1-V4; K_model =
       (b1 + b2) / (1 - a1 - a2) of the controller's model at [r, r, u_k-1, u_k-1], K_true = dCa/dTc of the cold
       steady state at u_k-1 with the plant parameters of that sample (central difference +-0.05 K);
    d. V5, configuration F: median over seeds of IAE(F) / IAE(K-MPC adaptive) <= 1.05 in each scenario; no constraint
       violations; fallback plan in at most 0.1 % of the steps.
Decision: passed if a-d hold. If only a fails, by one value not above 1.3, and b-d hold, the set with that value is
repeated once on seeds 40-59 (python validate_offset.py --repeat V<n>) and must satisfy a-c. Otherwise: not passed.

    python validate_offset.py                 # seeds 20-39, all sets; outputs/mpc_validation.json
    python validate_offset.py --repeat V2     # seeds 40-59, one set; outputs/mpc_validation_repeat.json
    python validate_offset.py --dev           # development check on used seeds (D1 0-11, S0 0-9), F and static only
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import mpc  # noqa: E402
import run_mpc as rm  # noqa: E402
import safety  # noqa: E402

FIXED = "K-MPC adaptive, excitation window"
CONFIGS = (FIXED, "K-MPC static", "K-MPC adaptive")
SEEDS, RESERVE = tuple(range(20, 40)), tuple(range(40, 60))
SETS = {"V1": ("S2",), "V2": ("D1",), "V3": ("Ta",), "V4": ("S2r",), "V5": ("S1", "S3", "S4", "S6")}
SCEN = {**cl.SCENARIOS, "D1": cl.scenario_d1, "Ta": cl.scenario_ta, "S2r": cl.scenario_s2r}
A_LIMIT, A_REPEAT = 1.0, 1.3          # 1e-3 mol/L
B_MEAN, B_SINGLE = 0.15, 0.5          # 1e-3 mol/L
C_GAMMA = 0.2
D_IAE_RATIO, D_FALLBACK = 1.05, 0.001
PROBE_EVERY, GAIN_STEP = 10, 0.05


class Probe:
    """Records gamma every PROBE_EVERY samples; everything else is delegated to the controller."""
    def __init__(self, inner, plant):
        self.inner, self.plant, self.gamma = inner, plant, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, k, y, u, r, info=None):
        if k % PROBE_EVERY == 0:
            u1 = float(u[k - 1])
            km = mpc.ss_gain(self.inner.model.g(np.array([r, r, u1, u1], float)))
            p = self.plant.params(k)
            kt = (safety.cold_steady_state(u1 + GAIN_STEP, p)[0] - safety.cold_steady_state(u1 - GAIN_STEP, p)[0]) \
                / (2 * GAIN_STEP)
            self.gamma.append((k, np.nan if km is None else km / kt))
        return self.inner.step(k, y, u, r, info)


def run_seed(args):
    """All runs of one seed: (seed, [(scenario, configuration)])."""
    seed, jobs = args
    out = []
    with threadpool_limits(1):
        lim = safety.tc_max()["tc_max"]
        c, block = rm.commissioning(seed)
        s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
        for scen, name in jobs:
            sc = SCEN[scen](seed, lim)
            ctrl, ident = rm.make(name, c, lim, s)
            probe = Probe(ctrl, sc["plant"]) if (name == FIXED and scen in ("S2", "D1", "Ta", "S2r")) else ctrl
            log = cl.run_loop(sc["plant"], sc["x0"], ident, probe, sc["r"], sc["add"], sc["missing"], sc["u0"],
                              supervisor=rm.CONFIGS[name][2], learn_after_change=rm.learn_after_change(name))
            m = cl.run_metrics(log, sc["starts"], safety.TC_MIN, lim, mpc.DU_MAX)
            hold = sc.get("hold_segments", [])
            out.append({"seed": seed, "scenario": scen, "config": name, "block": block,
                        "ss_error_1e3": [1e3 * m["segments"][i]["ss_error"] for i in hold],
                        "iae_total": m["iae_total"], "tc_total_variation": m["tc_total_variation"],
                        "violations": m["violations"], "fallback_steps": m["fallback_steps"], "n_steps": m["n_steps"],
                        "max_T": m["max_T"], "rows_added": int(getattr(ident, "n_rows_added", 0)),
                        "gamma": [g for _, g in probe.gamma] if isinstance(probe, Probe) else None})
    return out


def run(seeds, sets, configs, workers=rm.WORKERS):
    jobs = [(seed, [(scen, name) for st in sets for scen in SETS[st] for name in configs]) for seed in seeds]
    with ProcessPoolExecutor(workers) as ex:
        return [r for rows in ex.map(run_seed, jobs) for r in rows]


def get(rows, scen, name):
    return sorted([r for r in rows if r["scenario"] == scen and r["config"] == name], key=lambda r: r["seed"])


def evaluate(rows, sets):
    """Criteria a-d on the rows of the sets given (d only if V5 is among them)."""
    res = {"a": {"pass": True, "violations": []}, "b": {"pass": True, "segments": []},
           "c": {"pass": True, "gamma_min": None}, "d": {"pass": True, "scenarios": {}}}
    gmin = np.inf
    for st in sets:
        if st == "V5":
            continue
        for scen in SETS[st]:
            F, S = get(rows, scen, FIXED), get(rows, scen, "K-MPC static")
            for r in F:
                for i, e in enumerate(r["ss_error_1e3"]):
                    if abs(e) > A_LIMIT:
                        res["a"]["violations"].append({"set": st, "scenario": scen, "seed": r["seed"], "segment": i,
                                                       "error_1e3": e})
                g = np.array(r["gamma"], float)
                gmin = min(gmin, np.nanmin(g) if np.isfinite(g).any() else -np.inf)
                if np.isnan(g).any():
                    gmin = -np.inf
            if S:
                D = np.array([f["ss_error_1e3"] for f in F]) - np.array([s["ss_error_1e3"] for s in S])
                for i in range(D.shape[1]):
                    ok = abs(D[:, i].mean()) <= B_MEAN and np.max(np.abs(D[:, i])) <= B_SINGLE
                    res["b"]["segments"].append({"set": st, "scenario": scen, "segment": i,
                                                 "mean_diff_1e3": float(D[:, i].mean()),
                                                 "max_abs_diff_1e3": float(np.max(np.abs(D[:, i]))), "pass": bool(ok)})
                    res["b"]["pass"] &= bool(ok)
    res["a"]["pass"] = not res["a"]["violations"]
    res["c"]["gamma_min"] = None if gmin == np.inf else float(gmin)
    res["c"]["pass"] = gmin != np.inf and gmin >= C_GAMMA
    if "V5" in sets:
        for scen in SETS["V5"]:
            F, A = get(rows, scen, FIXED), get(rows, scen, "K-MPC adaptive")
            ratio = float(np.median([f["iae_total"] / a["iae_total"] for f, a in zip(F, A)]))
            viol = sum(f["violations"] for f in F)
            fb = sum(f["fallback_steps"] for f in F) / sum(f["n_steps"] for f in F)
            ok = ratio <= D_IAE_RATIO and viol == 0 and fb <= D_FALLBACK
            res["d"]["scenarios"][scen] = {"iae_ratio_median": ratio, "violations": viol, "fallback_share": fb,
                                           "pass": bool(ok)}
            res["d"]["pass"] &= bool(ok)
    return res


def decide(res):
    if all(res[k]["pass"] for k in "abcd"):
        return {"decision": "passed", "repeat": None}
    v = res["a"]["violations"]
    if all(res[k]["pass"] for k in "bcd") and len(v) == 1 and abs(v[0]["error_1e3"]) <= A_REPEAT:
        return {"decision": "repeat", "repeat": v[0]["set"]}
    return {"decision": "not passed", "repeat": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", choices=[k for k in SETS if k != "V5"])
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.dev:
        jobs = [(sd, [(scen, name) for scen in (("D1", "S0") if sd < 10 else ("D1",)) for name in (FIXED, "K-MPC static")])
                for sd in range(12)]
        with ProcessPoolExecutor(rm.WORKERS) as ex:
            rows = [r for rr in ex.map(run_seed, jobs) for r in rr]
        out = {"kind": "development check on used seeds (D1 0-11, S0 0-9)", "rows": rows}
        path = rm.OUT / "mpc_validation_dev.json"
        for scen in ("D1", "S0"):
            for name in (FIXED, "K-MPC static"):
                rr = [r for r in out["rows"] if r["scenario"] == scen and r["config"] == name]
                worst = [max(abs(e) for e in r["ss_error_1e3"]) for r in rr]
                print(f"{scen} {name}: {sum(w <= A_LIMIT for w in worst)} of {len(rr)} within {A_LIMIT}, worst "
                      f"{max(worst):.2f}", flush=True)
    elif args.repeat:
        rows = run(RESERVE, [args.repeat], CONFIGS)
        res = evaluate(rows, [args.repeat])
        ok = res["a"]["pass"] and res["b"]["pass"] and res["c"]["pass"]
        out = {"seeds": list(RESERVE), "set": args.repeat, "criteria": res, "passed": bool(ok), "rows": rows}
        path = rm.OUT / "mpc_validation_repeat.json"
        print("repeat passed" if ok else "repeat not passed", flush=True)
    else:
        rows = run(SEEDS, list(SETS), CONFIGS)
        res = evaluate(rows, list(SETS))
        out = {"seeds": list(SEEDS), "criteria": res, **decide(res), "rows": rows}
        path = rm.OUT / "mpc_validation.json"
        print(json.dumps({k: res[k]["pass"] for k in "abcd"}), out["decision"], out["repeat"], flush=True)
    out["wall_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
