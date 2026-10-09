"""One command: data -> start-up -> models -> validation -> report.

    python run_demo.py                  # simulated CSTR experiments, one seed (about 4 minutes on 2 cores)
    python run_demo.py --data log.csv   # your own log, columns: t_min, Tc_K, Ca_mol_L
    python run_seeds.py                 # the same over 20 seeds (about 37 minutes on 2 cores)

Everything is written to ./outputs (figures, metrics.json, report.md).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

import checks
import cstr
import data
import figures
import pipeline as pl
from adaptive import LAG, Supervisor, regressor, run_online
from report import write_report

OUT = Path(__file__).parent / "outputs"
TC_RANGE = (292.0, 303.0)       # coolant temperature range of the identification experiment, K
PROBE_TC = (292.0, 297.5, 303.0)  # inputs at which the GP variance is probed during the excursion
HOT_TC, HOT_LEN = 304.0, 60     # hot-side excursion: coolant temperature, K, and duration, samples
CA_MIN = 0.79                   # output constraint for a controller, mol/L: steady-state Ca at the upper
                                # edge of the range (0.787 at 303.0 K) rounded up; checked in process_facts
SENSITIVITY = {"nominal": {}, "UA -5 % (fouling)": {"UA": 0.95}, "UA -10 %": {"UA": 0.90},
               "Tf +1 K": {"Tf": +1.0}, "k0 x0.8 (end of the drift scenario)": {"k0": 0.8}}


def stationary(df, seed, figs=True):
    """Start-up on the first 5/8 of the log, test on the rest."""
    y, u = df["Ca_mol_L"].to_numpy(), df["Tc_K"].to_numpy()
    n_tr = int(0.625 * len(y))
    truth = "Ca_true" in df
    c = pl.commission(y[:n_tr], u[:n_tr], seed=seed)
    yte, ute, tte = y[n_tr:], u[n_tr:], df["t_min"].to_numpy()[n_tr:]
    y0 = pl.first_values(yte, c["y_clean"][-1])

    full = run_online(pl.identifier(c), yte, ute, y0)
    ref = df["Ca_true"].to_numpy()[n_tr:] if truth else full["y_clean"]
    models = {"ARX (linear, equation error)": pl.fit_arx(c), "OE (linear, output error)": pl.LinearOE(c),
              "Kernel ridge": pl.StaticKRR(c)}
    sims = {name: pl.free_run(m, ref, ute) for name, m in models.items()}
    Xte = np.array([regressor(full["y_clean"], ute, k) for k in range(LAG - 1, len(ute) - 1)])
    low = ref <= np.quantile(ref, 0.1)               # lowest tenth of Ca: hot, high-gain end
    krr = sims["Kernel ridge"]
    res = {"reference": "noise-free simulated output" if truth else "cleaned measurements",
           "n_startup": n_tr, "n_test": int(len(yte)), "sigma": c["sigma"], "lam": c["lam"],
           "noise_std_est": c["noise"],
           "models": {name: {"one_step_rmse": pl.rmse(m.predict(Xte), ref[LAG:]),
                             "free_run_rmse": pl.rmse(sims[name], ref),
                             "free_run_max_abs_err": float(np.max(np.abs(sims[name] - ref))),
                             "free_run_fit_percent": pl.fit_percent(ref, sims[name])}
                      for name, m in models.items()},
           "krr_free_run_rmse_low_ca": pl.rmse(krr, ref, low),
           "krr_free_run_rmse_elsewhere": pl.rmse(krr, ref, ~low),
           "ca_low_threshold": float(np.quantile(ref, 0.1)),
           "window_one_step_rmse": pl.rmse(full["pred"], ref),
           "filter": {"alpha": pl.ALPHA, "threshold": {str(a): v for a, v in c["thr"].items()},
                      "n_calibration_rows": c["n_calibration"],
                      "n_flagged_test": int(full["flagged"].sum())}}
    if truth:
        an = df["is_anomaly"].to_numpy()[n_tr:]
        nofilt = run_online(pl.identifier(c, use_filter=False), yte, ute, y0)
        res["filter"].update({
            "startup": pl.precision_recall(c["flagged"], df["is_anomaly"].to_numpy()[:n_tr]),
            "one_step_rmse_with_filter": pl.rmse(full["pred"], ref),
            "one_step_rmse_without_filter": pl.rmse(nofilt["pred"], ref),
            "sweep": []})
        dev = {name: data.rolling_median_dev(yte, causal=(name == "causal")) for name in ("causal", "centred")}
        for a in pl.ALPHAS:                      # every row: the full adaptive system
            o = full if a == pl.ALPHA else run_online(pl.identifier(c, alpha=a), yte, ute, y0)
            res["filter"]["sweep"].append({
                "alpha": a, "svr": pl.precision_recall(o["flagged"], an),
                **{f"median_{name}": pl.precision_recall(dev[name] > c["thr_median"][name][a], an)
                   for name in dev}})
    if figs:
        figures.filter_fig(tte, yte, full, OUT / "fig1_filter.png")
        figures.free_run_fig(tte, ref, sims, res["reference"], OUT / "fig2_free_run.png")
    res["checks"] = checks.run(c["Z"], c["tz"], c["sigma"], c["lam"], pl.SIGMAS, pl.LAMS, seed)
    res["checks"]["gradient_physical_max_rel_dev"] = checks.gradient_physical(models["Kernel ridge"], c["Xc"], seed)
    return res, c


def drift(seed, n0=2500, n1=3000, k0_end=0.8, figs=True):
    """Catalyst slowly loses activity: the rate constant falls linearly to k0_end.
    Two measures: one-step error, and free-run of each quarter by the model frozen at the
    start of that quarter (what a controller would rely on until the next update)."""
    rng = np.random.default_rng(seed + 10)
    Tc = data.aprbs(n0 + n1, *TC_RANGE, 10, 60, rng)
    scale = np.r_[np.ones(n0), np.linspace(1.0, k0_end, n1)]
    df = data.make_log(Tc, seed + 10, k0_scale=scale)
    y, u, true = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true"))
    c = pl.commission(y[:n0], u[:n0], seed=seed)
    yo, uo, to = y[n0:], u[n0:], true[n0:]
    y0 = pl.first_values(yo, c["y_clean"][-1])
    q = n1 // 4
    starts = (LAG, q, 2 * q, 3 * q)
    strategies = {"Static model, no filter": dict(use_filter=False, adapt=False),
                  "Static model + SVR filter": dict(use_filter=True, adapt=False),
                  "Sliding window, no filter": dict(use_filter=False, adapt=True),
                  "SVR filter + sliding window": dict(use_filter=True, adapt=True)}
    runs = {name: run_online(pl.identifier(c, **kw), yo, uo, y0, snapshot_at=starts)
            for name, kw in strategies.items()}
    an = df["is_anomaly"].to_numpy()[n0:]
    res = {"k0_end": k0_end, "n_startup": n0, "n_online": n1, "sigma": c["sigma"], "lam": c["lam"],
           "ca_shift_at_k0_end": {}, "strategies": {}}
    for name, o in runs.items():
        fr = []
        for i, s0 in enumerate(starts):
            seg = slice(i * q, (i + 1) * q)
            sim = pl.free_run(pl.Frozen(o["snapshots"][s0], c["scaler"]), to[seg], uo[seg])
            fr.append({"rmse": pl.rmse(sim, to[seg]), "max_abs_err": float(np.max(np.abs(sim - to[seg])))})
        res["strategies"][name] = {
            "rmse": pl.rmse(o["pred"], to),
            "rmse_by_quarter": [pl.rmse(o["pred"][i:i + q], to[i:i + q]) for i in range(0, n1, q)],
            "free_run_by_quarter": fr,
            "filter": pl.precision_recall(o["flagged"], an) if strategies[name]["use_filter"] else None}
    p_end = cstr.CSTRParams(k0=cstr.CSTRParams().k0 * k0_end)
    for Tc_ in (TC_RANGE[0], 297.0, TC_RANGE[1]):
        res["ca_shift_at_k0_end"][str(Tc_)] = float(cstr.steady_state(Tc_, p_end)[0] - cstr.steady_state(Tc_)[0])
    if figs:
        figures.drift_fig(np.arange(n1) * data.TS, to, runs, scale[n0:], OUT / "fig3_drift.png")
    return res


def _regime_stats(regime, start, end, H):
    """Samples from `start` until the first normal sample (after at least one non-normal),
    and how many samples before `start` / `end` the supervisor left normal (None if it had
    not been normal during the 5 H samples before)."""
    def lead(at):
        before = regime[max(at - 5 * H, 0):at][::-1]
        calm = np.where(before == 0)[0]
        return int(calm[0]) if len(calm) else None
    after = np.where(regime[start:end] > 0)[0]
    if len(after) == 0:
        back = None
    else:
        later = np.where(regime[start + after[0]:end] == 0)[0]
        back = int(after[0] + later[0]) if len(later) else None
    return {"samples_to_normal_after_leaving": back, "lead_before_leaving": lead(start),
            "lead_before_return": lead(end)}


def excursion(c, seed, figs=True):
    """The input leaves the range the model was trained on (cold side), then comes back."""
    rng = np.random.default_rng(seed + 20)
    parts = [(300, TC_RANGE), (500, (284.0, 290.0)), (400, TC_RANGE)]
    Tc = np.concatenate([data.aprbs(n, lo, hi, 10, 60, rng) for n, (lo, hi) in parts])
    df = data.make_log(Tc, seed + 20)
    y, u, true = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true"))
    y0 = pl.first_values(y, c["y_clean"][-1])
    runs = {"Static model": run_online(pl.identifier(c, adapt=False, horizon=pl.HORIZON), y, u, y0,
                                       probe_u=PROBE_TC),
            "Sliding window": run_online(pl.identifier(c, adapt=True, horizon=pl.HORIZON), y, u, y0,
                                         probe_u=PROBE_TC)}
    inside = np.r_[np.ones(300, bool), np.zeros(500, bool), np.ones(400, bool)]
    inside[:LAG] = False
    outside = ~inside
    outside[:LAG] = False
    res = {"outside_range": [284.0, 290.0], "th1": c["lam"], "th2": 3 * c["lam"], "horizon": pl.HORIZON,
           "probe_tc": list(PROBE_TC), "runs": {}}
    an = df["is_anomaly"].to_numpy()
    for name, o in runs.items():
        sup = Supervisor(c["lam"])                    # the same supervisor fed with the current point only
        reg_point = np.array([sup.update(v)[0] if np.isfinite(v) else 0 for v in o["var"]])
        res["runs"][name] = {
            "var_median_inside": float(np.nanmedian(o["var"][inside])),
            "var_median_outside": float(np.nanmedian(o["var"][outside])),
            "var_h_median_outside": float(np.nanmedian(o["var_h"][outside])),
            "probe_var_median_outside": np.nanmedian(o["probe_var"][outside], axis=0).tolist(),
            "regime_share_outside": (np.bincount(o["regime"][outside], minlength=3) / outside.sum()).tolist(),
            "regime_share_inside": (np.bincount(o["regime"][inside], minlength=3) / inside.sum()).tolist(),
            "point_only_regime_share_outside": (np.bincount(reg_point[outside], minlength=3)
                                                / outside.sum()).tolist(),
            "point_only": _regime_stats(reg_point, 300, 800, pl.HORIZON),
            "horizon": _regime_stats(o["regime"], 300, 800, pl.HORIZON),
            "mean_penalty_multiplier_outside": float(np.nanmean(o["mult"][outside])),
            "one_step_rmse_inside": pl.rmse(o["pred"], true, inside),
            "one_step_rmse_outside": pl.rmse(o["pred"], true, outside),
            "filter_outside": pl.precision_recall(o["flagged"] & outside, an & outside),
            "jump_samples_outside": int(o["jump"][outside].sum())}
    if figs:
        figures.supervisor_fig(np.arange(len(u)) * data.TS, u, runs, c["lam"], TC_RANGE,
                               OUT / "fig4_supervisor.png")
    return res


def hot_excursion(c, seed, figs=True, cold_limit=None):
    """The input exceeds the upper edge of the range by 1 K for HOT_LEN samples, past the
    point where the cold steady state disappears, and comes back."""
    rng = np.random.default_rng(seed + 30)
    Tc = np.r_[data.aprbs(300, *TC_RANGE, 10, 60, rng), np.full(HOT_LEN, HOT_TC),
               data.aprbs(440, *TC_RANGE, 10, 60, rng)]
    df = data.make_log(Tc, seed + 30)
    y, u, true, T = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true", "T_true"))
    y0 = pl.first_values(y, c["y_clean"][-1])
    runs = {"Static model": run_online(pl.identifier(c, adapt=False, horizon=pl.HORIZON), y, u, y0),
            "Sliding window": run_online(pl.identifier(c, adapt=True, horizon=pl.HORIZON), y, u, y0)}
    hot = slice(300, 300 + HOT_LEN)
    ca_lo = float(np.min(c["y_clean"]))
    below = np.where(true < ca_lo)[0]
    res = {"tc_hot": HOT_TC, "n_hot": HOT_LEN, "T_max": float(T.max()), "Ca_min": float(true.min()),
           "T_before": float(T[299]), "Ca_end": float(true[-1]), "T_end": float(T[-1]),
           "ca_min_startup": ca_lo,
           "first_sample_below_startup_ca": int(below[0]) if len(below) else None,
           "runs": {}}
    for name, o in runs.items():
        k0 = 300 - pl.HORIZON                       # the plan first contains the hot input here
        nonnormal = np.where(o["regime"][k0:] > 0)[0]
        res["runs"][name] = {
            "first_non_normal": int(nonnormal[0] + k0) if len(nonnormal) else None,
            "first_conservative": (int(np.where(o["regime"] == 2)[0][0])
                                   if np.any(o["regime"] == 2) else None),
            "var_h_before_ignition": float(np.nanmax(o["var_h"][300 - pl.HORIZON:300])),
            "one_step_rmse_last_200": pl.rmse(o["pred"][-200:], true[-200:])}
    if figs:
        figures.hot_fig(np.arange(len(u)) * data.TS, u, true, T, runs, c["lam"], TC_RANGE,
                        cold_limit, OUT / "fig5_hot_side.png")
    return res


def process_facts():
    """Steady-state facts about the simulated reactor across the identified range."""
    rows = []
    for Tc_ in (TC_RANGE[0], 297.0, 300.0, 302.0, TC_RANGE[1]):
        ss = cstr.steady_states(Tc_)
        ca0, T0 = ss[0]
        gain = (cstr.steady_states(Tc_ + 0.01)[0][0] - ca0) / 0.01
        eig = cstr.jacobian_eigenvalues([ca0, T0], Tc_)
        rows.append({"Tc": Tc_, "Ca": float(ca0), "T": float(T0), "gain": float(gain),
                     "slowest_time_constant_min": float(1.0 / np.min(-eig.real)),
                     "n_steady_states": len(ss)})
    p0 = cstr.CSTRParams()
    sens = []
    for name, ch in SENSITIVITY.items():
        p = replace(p0, **{k: (getattr(p0, k) + v if k == "Tf" else getattr(p0, k) * v) for k, v in ch.items()})
        lim = cstr.cold_branch_limit(p, lo=295.0, hi=315.0)
        ca, T = cstr.steady_states(lim - 1e-3, p)[0]
        sens.append({"case": name, "tc_limit": lim, "ca_at_limit": float(ca), "T_at_limit": float(T)})
    worst = max(r["ca_at_limit"] for r in sens)
    return {"cold_branch_limit_K": sens[0]["tc_limit"], "steady_states": rows, "limit_sensitivity": sens,
            "ca_min_constraint": CA_MIN, "ca_margin_to_worst_limit": CA_MIN - worst,
            "ca_constraint_above_data_edge": bool(CA_MIN >= rows[-1]["Ca"])}


def simulated_run(seed, figs=True, cold_limit=None):
    rng = np.random.default_rng(seed)
    df = data.make_log(data.aprbs(8000, *TC_RANGE, 10, 60, rng), seed, gap_at=2500)
    summary = {"seed": seed, "data": data.describe(df)}
    summary["stationary"], c = stationary(df, seed, figs)
    summary["drift"] = drift(seed, figs=figs)
    summary["excursion"] = excursion(c, seed, figs)
    summary["hot_excursion"] = hot_excursion(c, seed, figs, cold_limit)
    return summary, df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with columns t_min, Tc_K, Ca_mol_L (default: simulate)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    settings = {"window": pl.WINDOW, "refit_every": pl.REFIT_EVERY, "alpha": pl.ALPHA,
                "n_jump": 4, "lags": [2, 2], "horizon": pl.HORIZON, "tc_range": list(TC_RANGE)}
    if args.data:
        df = pd.read_csv(args.data)
        summary = {"simulated": False, "sampling_min": data.TS, "settings": settings, "data": data.describe(df)}
        summary["stationary"], _ = stationary(df, args.seed)
    else:
        process = process_facts()
        s, df = simulated_run(args.seed, cold_limit=process["cold_branch_limit_K"])
        df.to_csv(OUT / "raw_log.csv", index=False)
        summary = {"simulated": True, "sampling_min": data.TS, "settings": settings, **s,
                   "process": process}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    seeds = OUT / "seeds.json"
    write_report(summary, OUT / "report.md",
                 json.loads(seeds.read_text()) if seeds.exists() and not args.data else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
