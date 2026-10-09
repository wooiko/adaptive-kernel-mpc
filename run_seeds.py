"""Repeat the simulated experiments over many seeds and report the spread.

    python run_seeds.py              # seeds 0-19, all CPU cores (about 37 minutes on 2 cores)
    python run_seeds.py --n 30

Writes outputs/seeds.json and, if outputs/metrics.json exists (from run_demo.py),
rewrites outputs/report.md with the spread section.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cstr
import data
import pipeline as pl
import run_demo as rd
from report import write_report

OUT = Path(__file__).parent / "outputs"


def key_numbers(seed, cold_limit):
    s, _ = rd.simulated_run(seed, figs=False, cold_limit=cold_limit)
    st, dr, ex, hx = s["stationary"], s["drift"]["strategies"], s["excursion"], s["hot_excursion"]
    m, f = st["models"], st["filter"]
    sw = next(r for r in f["sweep"] if r["alpha"] == pl.ALPHA)
    krr = m["Kernel ridge"]["free_run_rmse"]
    w, a = ex["runs"]["Sliding window"], ex["runs"]["Static model"]
    hw, ha = hx["runs"]["Sliding window"], hx["runs"]["Static model"]
    warn = [r["first_non_normal"] for r in (hw, ha) if r["first_non_normal"] is not None]
    below = hx["first_sample_below_startup_ca"]
    return {
        "seed": seed,
        "noise_ratio": st["noise_std_est"] / data.NOISE_STD,
        "krr_free_run": krr,
        "arx_free_run": m["ARX (linear, equation error)"]["free_run_rmse"],
        "oe_free_run": m["OE (linear, output error)"]["free_run_rmse"],
        "arx_over_krr": m["ARX (linear, equation error)"]["free_run_rmse"] / krr,
        "oe_over_krr": m["OE (linear, output error)"]["free_run_rmse"] / krr,
        "svr_precision": sw["svr"]["precision"], "svr_recall": sw["svr"]["recall"],
        "svr_false_share": sw["svr"]["false_rejection_share"],
        "median_precision": sw["median_causal"]["precision"], "median_recall": sw["median_causal"]["recall"],
        "filter_gain": f["one_step_rmse_without_filter"] / f["one_step_rmse_with_filter"],
        "drift_one_step_ratio": dr["Static model, no filter"]["rmse"] / dr["SVR filter + sliding window"]["rmse"],
        "drift_filter_effect": dr["Static model, no filter"]["rmse"] / dr["Static model + SVR filter"]["rmse"],
        "drift_adapt_effect_q4": (dr["Static model + SVR filter"]["free_run_by_quarter"][3]["rmse"]
                                  / dr["SVR filter + sliding window"]["free_run_by_quarter"][3]["rmse"]),
        "exc_window_nonnormal": sum(w["regime_share_outside"][1:]),
        "exc_window_lead_return": w["horizon"]["lead_before_return"],
        "exc_window_back_to_normal": w["horizon"]["samples_to_normal_after_leaving"],
        "exc_static_nonnormal": sum(a["regime_share_outside"][1:]),
        "hot_var_before_over_th1": max(hw["var_h_before_ignition"], ha["var_h_before_ignition"]) / ex["th1"],
        "hot_warning_lag": (min(warn) - below) if warn and below is not None else None,
    }


def _job(args):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):            # one BLAS thread per worker: workers run in parallel instead
        return key_numbers(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of seeds (0 .. n-1)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    limit = cstr.cold_branch_limit()
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(_job, [(k, limit) for k in range(args.n)]), key=lambda r: r["seed"])
    res = {"n_seeds": args.n, "per_seed": rows}
    (OUT / "seeds.json").write_text(json.dumps(res, indent=2))
    metrics = OUT / "metrics.json"
    if metrics.exists():
        summary = json.loads(metrics.read_text())
        if summary.get("simulated"):
            write_report(summary, OUT / "report.md", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
