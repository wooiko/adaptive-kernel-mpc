"""Step time of K-MPC by the rule of TZ 14, D15 as refined on 13.10.2026 after the independent analysis of the step time
(the rule is fixed in the TZ before this measurement):

* measure: CPU time of the thread (time.thread_time) spent in the identifier and the controller on one sample, the wall
  time (time.perf_counter) of the same calls beside it; the intervals are those timed by closed_loop.run_loop
  (identifier: step, and learn/predict of a restored run; controller: step);
* for every K-MPC configuration separately, A = 95th percentile of all its samples in S1, S2, S3, S4, S5a, S6 over seeds
  0-19 together (as D11); the check is met if the median of A over three valid passes is <= 60 ms;
* a pass is valid if the lost time (wall minus CPU, summed over the timed intervals) is at most 5 % of the summed wall
  time and the machine speed benchmark (median of 150 fits of the KRR and of 150 of the SVR filter on the same 1000 rows)
  differs by at most 10 % between before and after the pass; an invalid pass is repeated at most twice;
* samples with a refit of the window model are part of A; their share and their time are reported apart.

Separate run, one process pinned to one vCPU (the other one left to the system), one BLAS thread; the non-time results
of every run are compared with outputs/mpc_seeds.json (same computation). Results: outputs/mpc_step_time.json; the
samples of each pass: outputs/step_time/pass<i>.npz (float32).

    python measure_step_time.py              # 3 valid passes, about 4 h each on the cloud machine of the report
    python measure_step_time.py --passes 1 --seeds 2   # short check of the procedure (not a result)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import mpc  # noqa: E402
import run_mpc as rm  # noqa: E402
import safety  # noqa: E402
from adaptive import fit_svr  # noqa: E402
from kernel import KRR  # noqa: E402

LIMIT_MS = 60.0          # TZ 4: 1 % of the sampling period
PCT = 95.0
N_PASSES = 3             # D15: median of three valid passes
MAX_REPEATS = 2          # an invalid pass is repeated at most twice, then the measurement stops (P14)
LOST_MAX = 0.05          # share of the wall time lost to preemption (analysis, M4: 0.35-1.79 % alone, 27-33 % overloaded)
BENCH_DRIFT_MAX = 0.10   # relative change of the speed benchmark between before and after a pass
BENCH_REPEATS = 150       # fits of each kind per benchmark (30 gave 55.1-63.5 ms in 8 repeats, 150 gave 56.0-57.1)
IN_SET = ("S1", "S2", "S3", "S4", "S5a", "S6")
OUT = rm.OUT / "mpc_step_time.json"
SAMPLES = rm.OUT / "step_time"
REF = "K-MPC adaptive, reference without spikes"


def kmpc_jobs(seeds):
    """(seed, scenario, configuration) of every K-MPC run inside the admissible set (run_mpc.PLAN)."""
    return [(seed, scen, name) for seed in seeds for scen in IN_SET for name in rm.PLAN[scen]
            if name.startswith("K-MPC")]


class Clock:
    """CPU and wall time of the identifier and the controller per sample, and whether the window model was refitted."""
    def __init__(self):
        self.rows, self.cur = [], None

    def new_sample(self):
        self.cur = [0.0, 0.0, 0.0, 0.0, 0.0]          # ident cpu, ident wall, ctrl cpu, ctrl wall, refit
        self.rows.append(self.cur)

    def timed(self, part, fn):
        def wrapped(*a, **kw):
            if part == "ident" and fn.__name__ == "step":
                self.new_sample()
            c0, w0 = time.thread_time(), time.perf_counter()
            out = fn(*a, **kw)
            c1, w1 = time.thread_time(), time.perf_counter()
            i = 0 if part == "ident" else 2
            self.cur[i] += 1e3 * (c1 - c0)
            self.cur[i + 1] += 1e3 * (w1 - w0)
            return out
        return wrapped

    def mark_refit(self):
        if self.cur is not None:
            self.cur[4] = 1.0


class TimedIdent:
    """The identifier as run_loop sees it: step, learn and predict timed; every other attribute passed through."""
    def __init__(self, inner, clock):
        object.__setattr__(self, "_i", inner)
        object.__setattr__(self, "_c", clock)

    def __getattr__(self, name):
        a = getattr(self._i, name)
        return self._c.timed("ident", a) if name in ("step", "learn", "predict") else a

    def __setattr__(self, name, value):
        setattr(self._i, name, value)


class TimedCtrl:
    def __init__(self, inner, clock):
        self.inner, self.clock = inner, clock

    def planned_inputs(self, u_prev, n):
        return self.inner.planned_inputs(u_prev, n)

    def step(self, k, y, u, r, info):
        return self.clock.timed("ctrl", self.inner.step)(k, y, u, r, info)


def watch_refits(ident, clock):
    """Mark the samples on which the window model is refitted (adaptive.AdaptiveIdentifier._refit)."""
    orig = ident._refit

    def refit():
        clock.mark_refit()
        return orig()
    ident._refit = refit


def steal_ticks():
    try:
        return int(Path("/proc/stat").read_text().split("\n")[0].split()[8])
    except (OSError, IndexError, ValueError):
        return None


def measured_run(c, seed, lim, scen, name, s):
    sc = cl.scenario_s4(seed, lim, spikes=False) if name == REF else cl.SCENARIOS[scen](seed, lim)
    ctrl, ident = rm.make(name, c, lim, s)
    clock = Clock()
    watch_refits(ident, clock)
    ru0, st0 = resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw, steal_ticks()
    log = cl.run_loop(sc["plant"], sc["x0"], TimedIdent(ident, clock), TimedCtrl(ctrl, clock), sc["r"], sc["add"],
                      sc["missing"], sc["u0"], supervisor=rm.CONFIGS[name][2],
                      learn_after_change=rm.learn_after_change(name))
    ru1, st1 = resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw, steal_ticks()
    m = cl.run_metrics(log, sc["starts"], safety.TC_MIN, lim, mpc.DU_MAX)
    return np.array(clock.rows, float), m, {"involuntary_switches": ru1 - ru0,
                                            "steal_ticks": None if st0 is None else st1 - st0}


def benchmark(c):
    """Machine speed: median time of BENCH_REPEATS fits of the KRR and of the SVR filter on the same 1000 rows."""
    Z, tz = np.asarray(c["Z"])[-1000:], np.asarray(c["tz"])[-1000:]
    noise_z = c["noise"] / c["scaler"].st
    out = {}
    for key, fn in (("krr_ms", lambda: KRR(c["sigma"], c["lam"]).fit(Z, tz)),
                    ("svr_ms", lambda: fit_svr(Z, tz, c["sigma"], noise_z))):
        t = []
        for _ in range(BENCH_REPEATS):
            t0 = time.thread_time()
            fn()
            t.append(1e3 * (time.thread_time() - t0))
        out[key] = float(np.median(t))
    out["total_ms"] = out["krr_ms"] + out["svr_ms"]
    return out


TIME_KEYS = ("step_ms", "ident_ms", "full_ms", "wall_s", "wall_s_total")


def no_time(o):
    if isinstance(o, dict):
        return {k: no_time(v) for k, v in o.items() if k not in TIME_KEYS}
    if isinstance(o, list):
        return [no_time(v) for v in o]
    return o


def pooled(rows_by_conf):
    """Per configuration: A (CPU p95 of all samples), B (wall p95), refit share and the time of the refit samples."""
    out = {}
    for name, R in rows_by_conf.items():
        R = np.vstack(R)
        cpu, wall, ref = R[:, 0] + R[:, 2], R[:, 1] + R[:, 3], R[:, 4] > 0
        out[name] = {"samples": int(len(R)), "A_cpu_p95_ms": float(np.percentile(cpu, PCT)),
                     "B_wall_p95_ms": float(np.percentile(wall, PCT)),
                     "cpu_p95_without_refit_ms": float(np.percentile(cpu[~ref], PCT)),
                     "refit_share": float(ref.mean()),
                     "refit_cpu_median_ms": float(np.median(cpu[ref])) if ref.any() else None,
                     "refit_cpu_p95_ms": float(np.percentile(cpu[ref], PCT)) if ref.any() else None,
                     "refit_samples_over_limit_cpu": int(np.sum(cpu[ref] > LIMIT_MS)),
                     "samples_over_limit_cpu": int(np.sum(cpu > LIMIT_MS))}
    return out


def pass_valid(lost_share, bench_before, bench_after):
    """D15: lost time at most LOST_MAX of the wall time, speed benchmark changed by at most BENCH_DRIFT_MAX."""
    drift = abs(bench_after - bench_before) / bench_before
    return bool(lost_share <= LOST_MAX and drift <= BENCH_DRIFT_MAX), float(drift)


def decide(passes, limit=LIMIT_MS):
    """D15: per configuration, the median of A over the valid passes, and whether it is <= limit."""
    valid = [p for p in passes if p["valid"]]
    out = {}
    for name in valid[0]["configs"] if valid else []:
        a = [p["configs"][name]["A_cpu_p95_ms"] for p in valid]
        med = float(np.median(a))
        out[name] = {"A_by_pass_ms": a, "A_median_ms": med, "pass": bool(med <= limit)}
    return out


def machine_full():
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().split("\n"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    virt = None
    try:
        virt = "hypervisor" in Path("/proc/cpuinfo").read_text()
    except OSError:
        pass
    return {"platform": platform.platform(), "cpu_model": cpu, "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "hypervisor_flag": virt, "python": platform.python_version(),
            "blas": [{k: d.get(k) for k in ("internal_api", "version", "num_threads", "filepath")}
                     for d in threadpool_info()]}


def one_pass(comm, jobs, lim, s, ref_results, bench_c, label):
    bench0 = benchmark(bench_c)
    t0 = time.time()
    rows, runs, mism = {}, [], 0
    for seed, scen, name in jobs:
        R, m, sysinfo = measured_run(comm[seed], seed, lim, scen, name, s)
        ref = ref_results[str(seed)]["results"][scen][name]
        same = no_time(m) == no_time({k: ref[k] for k in m if k in ref})
        mism += not same
        rows.setdefault(name, []).append(R)
        runs.append({"seed": seed, "scenario": scen, "config": name, "samples": int(len(R)),
                     "refits": int(R[:, 4].sum()), "same_as_mpc_seeds": same, **sysinfo})
    bench1 = benchmark(bench_c)
    allr = np.vstack([r for v in rows.values() for r in v])
    wall, cpu = allr[:, 1].sum() + allr[:, 3].sum(), allr[:, 0].sum() + allr[:, 2].sum()
    lost = float((wall - cpu) / wall)
    ok, drift = pass_valid(lost, bench0["total_ms"], bench1["total_ms"])
    SAMPLES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAMPLES / f"{label}.npz", **{n.replace(" ", "_").replace(",", ""): np.vstack(v).astype(np.float32)
                                                    for n, v in rows.items()})
    return {"label": label, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": time.time() - t0,
            "benchmark_before": bench0, "benchmark_after": bench1, "benchmark_drift": drift, "lost_share": lost,
            "valid": ok, "runs_not_equal_to_mpc_seeds": int(mism), "configs": pooled(rows), "runs": runs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=int, default=N_PASSES)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if hasattr(os, "sched_setaffinity") and (os.cpu_count() or 1) > 1:
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    lim = safety.tc_max()["tc_max"]
    s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
    ref_results = json.loads((rm.OUT / "mpc_seeds.json").read_text())["by_seed"]
    seeds = list(range(args.seeds))
    with threadpool_limits(1):
        comm = {seed: rm.commissioning(seed)[0] for seed in seeds}
        jobs = kmpc_jobs(seeds)
        passes, tries = [], 0
        while sum(p["valid"] for p in passes) < args.passes:
            if tries >= args.passes + MAX_REPEATS:
                break
            tries += 1
            p = one_pass(comm, jobs, lim, s, ref_results, comm[0], f"pass{tries}")
            passes.append(p)
            print(f"pass {tries}: valid {p['valid']} (lost {100 * p['lost_share']:.2f} %, benchmark drift "
                  f"{100 * p['benchmark_drift']:.1f} %), {p['wall_s']:.0f} s", flush=True)
            args.out.write_text(json.dumps(result(passes, args, seeds), indent=1))
    res = result(passes, args, seeds)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v["A_median_ms"] for k, v in res["decision"].items()}, indent=1))


def result(passes, args, seeds):
    complete = sum(p["valid"] for p in passes) >= args.passes
    return {"rule": "TZ 14, D15 (refined 13.10.2026): per K-MPC configuration, A = p95 of the CPU time of identifier + "
                    f"controller over all samples of {', '.join(IN_SET)}, seeds {seeds[0]}-{seeds[-1]}; met if the median "
                    f"of A over {args.passes} valid passes <= {LIMIT_MS:g} ms; valid pass: lost time <= {100 * LOST_MAX:g} %, "
                    f"benchmark drift <= {100 * BENCH_DRIFT_MAX:g} %",
            "complete": complete, "machine": machine_full(), "provenance": rm.provenance(0),
            "decision": decide(passes) if complete else {},
            "passes": [{k: v for k, v in p.items()} for p in passes]}


if __name__ == "__main__":
    main()
