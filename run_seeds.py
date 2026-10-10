"""Repeat the simulated experiments over many seeds and report the spread.

    python run_seeds.py              # seeds 0-19, all CPU cores (about 60 minutes on 2 cores)
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
    below = hx["first_sample_below_startup_ca"]

    def lag(r):
        """First warning relative to Ca leaving the start-up range (negative = before); None if
        no warning, or if the supervisor was already not normal before the hot input could be seen."""
        ok = r["normal_before"] and r["first_non_normal"] is not None and below is not None
        return r["first_non_normal"] - below if ok else None

    return {
        "seed": seed,
        "noise_ratio": st["noise_std_est"] / data.NOISE_STD,
        "sigma": st["sigma"], "lam": st["lam"],
        "tune_n_candidates": st["tuning"]["n_candidates_loo"], "tune_n_admissible": st["tuning"]["n_admissible"],
        "tune_loo_min_free_run": st["tuning"]["loo_minimum"]["holdout_free_run_rmse"],
        "tune_chosen_free_run": st["tuning"]["chosen"]["holdout_free_run_rmse"],
        "tune_worst_candidate_free_run": st["tuning"]["holdout_free_run_rmse_range"][1],
        "tune_best_candidate_free_run": st["tuning"]["holdout_free_run_rmse_range"][0],
        "tune_chosen_over_best": (st["tuning"]["chosen"]["holdout_free_run_rmse"]
                                  / st["tuning"]["holdout_free_run_rmse_range"][0]),
        "lam_over_noise_var": st["lam"] / st["noise_var_normalised"],
        "tune_changed_by_free_run": st["tuning"]["changed_by_free_run"],
        "tune_loo_min_on_edge": st["tuning"]["loo_min_on_grid_edge"],
        "tune_test_worst_admissible": st["tuning_test_check"]["worst_admissible"],
        "tune_test_best_admissible": st["tuning_test_check"]["best_admissible"],
        "krr_free_run": krr,
        "arx_free_run": m["ARX (linear, equation error)"]["free_run_rmse"],
        "oe_free_run": m["OE (linear, output error)"]["free_run_rmse"],
        "arx_over_krr": m["ARX (linear, equation error)"]["free_run_rmse"] / krr,
        "oe_over_krr": m["OE (linear, output error)"]["free_run_rmse"] / krr,
        "svr_precision": sw["svr"]["precision"], "svr_recall": sw["svr"]["recall"],
        "svr_false_share": sw["svr"]["false_rejection_share"],
        "svr_precision_final": sw["svr_final"]["precision"], "svr_recall_final": sw["svr_final"]["recall"],
        "svr_false_share_final": sw["svr_final"]["false_rejection_share"],
        "median_precision": sw["median_causal"]["precision"], "median_recall": sw["median_causal"]["recall"],
        "filter_gain": f["one_step_rmse_without_filter"] / f["one_step_rmse_with_filter"],
        "drift_one_step_ratio": dr["Static model, no filter"]["rmse"] / dr["SVR filter + sliding window"]["rmse"],
        "drift_filter_effect": dr["Static model, no filter"]["rmse"] / dr["Static model + SVR filter"]["rmse"],
        "drift_adapt_effect_q4": (dr["Static model + SVR filter"]["free_run_by_quarter"][3]["rmse"]
                                  / dr["SVR filter + sliding window"]["free_run_by_quarter"][3]["rmse"]),
        "exc_window_nonnormal": sum(w["regime_share_outside"][1:]),
        "exc_window_inside_nonnormal": sum(w["regime_share_inside"][1:]),
        "exc_window_lead_return": w["horizon"]["lead_before_return"],
        "exc_max_lead": ex["max_lead"],
        "exc_window_back_to_normal": w["horizon"]["samples_to_normal_after_leaving"],
        "exc_static_nonnormal": sum(a["regime_share_outside"][1:]),
        "hot_var_before_over_th1_window": hw["var_h_before_ignition"] / ex["th1"],
        "hot_var_before_over_th1_static": ha["var_h_before_ignition"] / ex["th1"],
        "hot_hidden_window": len(hw["hidden_in_real_time"]),
        "hot_hidden_max_overstatement_window": hw["hidden_max_overstatement"],
        "sup_test_share_not_normal": st["supervisor_test"]["share_not_normal"],
        "sup_test_changes": st["supervisor_test"]["n_input_changes"],
        "sup_test_warned_first": st["supervisor_test"]["n_warned_first_opportunity"],
        "hot_warning_lag_window": lag(hw), "hot_warning_lag_static": lag(ha),
        "hot_normal_before_window": hw["normal_before"], "hot_normal_before_static": ha["normal_before"],
        "hot_first_opportunity": hx["first_opportunity"], "hot_below": below,
        "hot_first_warning_window": hw["first_non_normal"], "hot_first_warning_static": ha["first_non_normal"],
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
