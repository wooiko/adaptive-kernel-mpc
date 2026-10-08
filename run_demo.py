"""One command: data -> start-up -> models -> validation -> report.

    python run_demo.py                  # simulated CSTR experiments (about 2 minutes)
    python run_demo.py --data log.csv   # your own log, columns: t_min, Tc_K, Ca_mol_L

Everything is written to ./outputs (figures, metrics.json, report.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import checks
import data
import figures
import pipeline as pl
from adaptive import LAG, regressor, run_online
from report import write_report

OUT = Path(__file__).parent / "outputs"
TC_RANGE = (292.0, 303.0)       # coolant temperature range of the identification experiment, K


def stationary(df, seed):
    """Start-up on the first 5/8 of the log, test on the rest."""
    y, u = df["Ca_mol_L"].to_numpy(), df["Tc_K"].to_numpy()
    n_tr = int(0.625 * len(y))
    truth = "Ca_true" in df
    c = pl.commission(y[:n_tr], u[:n_tr], seed=seed)
    yte, ute, tte = y[n_tr:], u[n_tr:], df["t_min"].to_numpy()[n_tr:]

    full = run_online(pl.identifier(c), yte, ute, c["y_clean"][-LAG:])
    ref = df["Ca_true"].to_numpy()[n_tr:] if truth else full["y_clean"]
    krr, arx = pl.StaticKRR(c), pl.fit_arx(c)
    sims = {"ARX (linear)": pl.free_run(arx, ref, ute), "Kernel ridge": pl.free_run(krr, ref, ute)}
    Xte = np.array([regressor(full["y_clean"], ute, k) for k in range(LAG - 1, len(ute) - 1)])
    low = ref <= np.quantile(ref, 0.1)               # lowest tenth of Ca: hot, high-gain end
    res = {"reference": "noise-free simulated output" if truth else "cleaned measurements",
           "n_startup": n_tr, "n_test": int(len(yte)), "sigma": c["sigma"], "lam": c["lam"],
           "noise_std_est": c["noise"],
           "models": {name: {"one_step_rmse": pl.rmse(m.predict(Xte), ref[LAG:]),
                             "free_run_rmse": pl.rmse(sims[name], ref),
                             "free_run_max_abs_err": float(np.max(np.abs(sims[name] - ref))),
                             "free_run_fit_percent": pl.fit_percent(ref, sims[name])}
                      for name, m in (("ARX (linear)", arx), ("Kernel ridge", krr))},
           "krr_free_run_rmse_low_ca": pl.rmse(sims["Kernel ridge"], ref, low),
           "krr_free_run_rmse_elsewhere": pl.rmse(sims["Kernel ridge"], ref, ~low),
           "ca_low_threshold": float(np.quantile(ref, 0.1)),
           "window_one_step_rmse": pl.rmse(full["pred"], ref),
           "filter": {"n_flagged_test": int(full["flagged"].sum())}}
    if truth:
        an = df["is_anomaly"].to_numpy()[n_tr:]
        nofilt = run_online(pl.identifier(c, use_filter=False), yte, ute, c["y_clean"][-LAG:])
        res["filter"].update({
            "startup": pl.precision_recall(c["flagged"], df["is_anomaly"].to_numpy()[:n_tr]),
            "one_step_rmse_with_filter": pl.rmse(full["pred"], ref),
            "one_step_rmse_without_filter": pl.rmse(nofilt["pred"], ref),
            "sweep": []})
        for kappa in (2.5, 3.5, 4.5):
            o = full if kappa == pl.KAPPA else run_online(
                pl.identifier(c, kappa=kappa, adapt=False), yte, ute, c["y_clean"][-LAG:])
            res["filter"]["sweep"].append({
                "kappa": kappa, "svr": pl.precision_recall(o["flagged"], an),
                "rolling_median": pl.precision_recall(
                    data.rolling_median_flags(yte, kappa, c["noise"]), an)})
    figures.filter_fig(tte, yte, full, OUT / "fig1_filter.png")
    figures.free_run_fig(tte, ref, sims, res["reference"], OUT / "fig2_free_run.png")
    res["checks"] = checks.run(c["Z"], c["tz"], c["sigma"], c["lam"], pl.SIGMAS, pl.LAMS, seed)
    return res, c


def drift(seed, n0=2500, n1=3000, k0_end=0.8):
    """Catalyst slowly loses activity: the rate constant falls linearly to k0_end."""
    rng = np.random.default_rng(seed + 10)
    Tc = data.aprbs(n0 + n1, *TC_RANGE, 10, 60, rng)
    scale = np.r_[np.ones(n0), np.linspace(1.0, k0_end, n1)]
    df = data.make_log(Tc, seed + 10, k0_scale=scale)
    y, u, true = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true"))
    c = pl.commission(y[:n0], u[:n0], seed=seed)
    yo, uo, to = y[n0:], u[n0:], true[n0:]
    strategies = {"Static model, no filter": dict(use_filter=False, adapt=False),
                  "Sliding window, no filter": dict(use_filter=False, adapt=True),
                  "SVR filter + sliding window": dict(use_filter=True, adapt=True)}
    runs = {name: run_online(pl.identifier(c, **kw), yo, uo, c["y_clean"][-LAG:])
            for name, kw in strategies.items()}
    q = n1 // 4
    res = {"k0_end": k0_end, "n_startup": n0, "n_online": n1, "sigma": c["sigma"], "lam": c["lam"],
           "strategies": {name: {"rmse": pl.rmse(o["pred"], to),
                                 "rmse_by_quarter": [pl.rmse(o["pred"][i:i + q], to[i:i + q])
                                                     for i in range(0, n1, q)]}
                          for name, o in runs.items()},
           "filter": pl.precision_recall(runs["SVR filter + sliding window"]["flagged"],
                                         df["is_anomaly"].to_numpy()[n0:])}
    figures.drift_fig(np.arange(n1) * data.TS, to, runs, scale[n0:], OUT / "fig3_drift.png")
    return res


def excursion(c, seed):
    """The input leaves the range the model was trained on, then comes back."""
    rng = np.random.default_rng(seed + 20)
    parts = [(300, TC_RANGE), (500, (284.0, 290.0)), (400, TC_RANGE)]
    Tc = np.concatenate([data.aprbs(n, lo, hi, 10, 60, rng) for n, (lo, hi) in parts])
    df = data.make_log(Tc, seed + 20)
    y, u, true = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true"))
    runs = {"Static model": run_online(pl.identifier(c, adapt=False), y, u, true[:LAG]),
            "Sliding window": run_online(pl.identifier(c, adapt=True), y, u, true[:LAG])}
    inside = np.r_[np.ones(300, bool), np.zeros(500, bool), np.ones(400, bool)]
    inside[:LAG] = False
    outside = ~inside
    outside[:LAG] = False
    res = {"outside_range": [284.0, 290.0], "th1": c["lam"], "th2": 3 * c["lam"], "runs": {}}
    for name, o in runs.items():
        res["runs"][name] = {
            "var_median_inside": float(np.nanmedian(o["var"][inside])),
            "var_median_outside": float(np.nanmedian(o["var"][outside])),
            "regime_share_outside": (np.bincount(o["regime"][outside], minlength=3)
                                     / outside.sum()).tolist(),
            "regime_share_inside": (np.bincount(o["regime"][inside], minlength=3)
                                    / inside.sum()).tolist(),
            "mean_penalty_multiplier_outside": float(np.nanmean(o["mult"][outside])),
            "one_step_rmse_inside": pl.rmse(o["pred"], true, inside),
            "one_step_rmse_outside": pl.rmse(o["pred"], true, outside)}
    figures.supervisor_fig(np.arange(len(u)) * data.TS, u, runs, c["lam"], TC_RANGE,
                           OUT / "fig4_supervisor.png")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with columns t_min, Tc_K, Ca_mol_L (default: simulate)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.data:
        df = pd.read_csv(args.data)
    else:
        rng = np.random.default_rng(args.seed)
        df = data.make_log(data.aprbs(8000, *TC_RANGE, 10, 60, rng), args.seed, gap_at=2500)
        df.to_csv(OUT / "raw_log.csv", index=False)

    summary = {"simulated": args.data is None, "sampling_min": data.TS,
               "settings": {"window": pl.WINDOW, "refit_every": pl.REFIT_EVERY, "kappa": pl.KAPPA,
                            "n_jump": 4, "lags": [2, 2]},
               "data": data.describe(df)}
    summary["stationary"], c = stationary(df, args.seed)
    if args.data is None:
        summary["drift"] = drift(args.seed)
        summary["excursion"] = excursion(c, args.seed)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, OUT / "report.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
