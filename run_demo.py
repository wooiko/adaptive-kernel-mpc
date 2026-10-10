"""One command: data -> start-up -> models -> validation -> report.

    python run_demo.py                  # simulated CSTR experiments, one seed (about 6 minutes)
    python run_demo.py --data log.csv   # your own log, columns: t_min, Tc_K, Ca_mol_L
    python run_seeds.py                 # the same over 20 seeds (about 60 minutes on 2 cores)

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
from threadpoolctl import threadpool_limits

from adaptive import LAG, N_JUMP, NA, NB, REFIT_EVERY_JUMP, Supervisor, first_opportunity, regressor, run_online
from report import write_report

OUT = Path(__file__).parent / "outputs"
TC_RANGE = (292.0, 303.0)       # coolant temperature range of the identification experiment, K
PROBE_TC = (292.0, 297.5, 303.0)  # inputs at which the GP variance is probed during the excursion
HOT_TC, HOT_LEN = 304.0, 60     # hot-side excursion: coolant temperature, K, and duration, samples
HOT_START, HOT_TOTAL = 300, 800  # hot-side excursion: sample of the step to HOT_TC; record length
EXC_PARTS = (300, 500, 400)     # cold excursion: samples inside, outside, inside the range
STARTUP_SHARE = 0.625           # first share of a log used for start-up; the rest is the test
CA_MIN = 0.79                   # candidate output constraint, mol/L: steady-state Ca at the upper edge of
                                # the range (0.787 at 303.0 K) rounded up; tested in process_facts
SENSITIVITY = {"nominal": {}, "UA -5 % (fouling)": {"UA": 0.95}, "UA -10 %": {"UA": 0.90},
               "Tf +1 K": {"Tf": +1.0}, "k0 x0.8 (end of the drift scenario)": {"k0": 0.8},
               "Caf +5 %": {"Caf": 1.05}, "Caf -5 %": {"Caf": 0.95},
               "q +10 %": {"q": 1.10}, "q -10 %": {"q": 0.90},
               "UA -5 % and Caf +5 %": {"UA": 0.95, "Caf": 1.05}}
IGNITION_CASES = {"UA -10 %": {"UA": 0.90}, "Caf +5 %": {"Caf": 1.05}}   # limit inside the range
IGNITION_MARGIN_K = 0.5         # the ignition test starts on the cold steady state this far below the limit
DELAYS_MIN = (0.0, 0.3, 0.5, 1.0)  # delay between Ca crossing CA_MIN and full cooling, min
SCENARIO = {"stationary": 0, "drift": 1, "excursion": 2, "hot": 3}       # random-stream ids
MIN_LOG = int(np.ceil((pl.MIN_CAL_ROWS + pl.WINDOW + LAG) / STARTUP_SHARE))  # necessary, not sufficient:
                                # rows with spikes or gaps do not count for calibration (pipeline.startup_check)


def params(change, p0=cstr.CSTRParams()):
    """Process parameters with relative changes (Tf: additive, K)."""
    return replace(p0, **{k: (getattr(p0, k) + v if k == "Tf" else getattr(p0, k) * v) for k, v in change.items()})


def stationary(df, seed, figs=True):
    """Start-up on the first 5/8 of the log, test on the rest."""
    y, u = df["Ca_mol_L"].to_numpy(), df["Tc_K"].to_numpy()
    n_tr = int(STARTUP_SHARE * len(y))
    truth = "Ca_true" in df
    c = pl.commission(y[:n_tr], u[:n_tr], seed=seed)
    yte, ute, tte = y[n_tr:], u[n_tr:], df["t_min"].to_numpy()[n_tr:]
    y0 = pl.first_values(yte, c["y_clean"][-1])

    full = run_online(pl.identifier(c, horizon=pl.HORIZON), yte, ute, y0)   # the horizon changes only var_h / regime
    ref = df["Ca_true"].to_numpy()[n_tr:] if truth else full["y_clean"]
    models = {"ARX (linear, equation error)": pl.fit_arx(c), "OE (linear, output error)": pl.LinearOE(c),
              "Kernel ridge": pl.StaticKRR(c)}
    sims = {name: pl.free_run(m, ref, ute) for name, m in models.items()}
    Xte = np.array([regressor(full["y_clean"], ute, k) for k in range(LAG - 1, len(ute) - 1)])
    low = ref <= np.quantile(ref, 0.1)               # lowest tenth of Ca: hot, high-gain end
    krr = sims["Kernel ridge"]
    res = {"reference": "noise-free simulated output" if truth else "cleaned measurements",
           "n_startup": n_tr, "n_test": int(len(yte)), "sigma": c["sigma"], "lam": c["lam"],
           "tuning": c["tuning"], "noise_std_est": c["noise"],
           "noise_var_normalised": float((c["noise"] / c["scaler"].st) ** 2),
           "models": {name: {"one_step_rmse": pl.rmse(m.predict(Xte), ref[LAG:]),
                             "free_run_rmse": pl.rmse(sims[name], ref),
                             "free_run_max_abs_err": float(np.max(np.abs(sims[name] - ref))),
                             "free_run_fit_percent": pl.fit_percent(ref, sims[name])}
                      for name, m in models.items()},
           "krr_free_run_rmse_low_ca": pl.rmse(krr, ref, low),
           "krr_free_run_rmse_elsewhere": pl.rmse(krr, ref, ~low),
           "ca_low_threshold": float(np.quantile(ref, 0.1)),
           "filter": {"alpha": pl.ALPHA, "threshold": {str(a): v for a, v in c["thr"].items()},
                      "n_calibration_rows": c["n_calibration"],
                      "n_flagged_test": int(full["flagged_rt"].sum()),
                      "n_flagged_test_final": int(full["flagged"].sum())}}
    if truth:
        # diagnostic only, not part of the selection (it uses the test truth): the test free run of every
        # admissible tuning candidate, each trained like the static model on all start-up rows
        adm = [pl.rmse(pl.free_run(pl.StaticKRR(c, sg, lm), ref, ute), ref) for sg, lm in c["tuning"]["admissible"]]
        res["tuning_test_check"] = {"chosen": res["models"]["Kernel ridge"]["free_run_rmse"],
                                    "worst_admissible": float(max(adm)), "best_admissible": float(min(adm)),
                                    "n_admissible": len(adm)}
        an = df["is_anomaly"].to_numpy()[n_tr:]
        nofilt = run_online(pl.identifier(c, use_filter=False), yte, ute, y0)
        res["filter"].update({
            "one_step_rmse_with_filter": pl.rmse(full["pred"], ref),
            "one_step_rmse_without_filter": pl.rmse(nofilt["pred"], ref),
            "sweep": []})
        dev = {name: data.rolling_median_dev(yte, causal=(name == "causal")) for name in ("causal", "centred")}
        for a in pl.ALPHAS:                      # every row: the full adaptive system
            o = full if a == pl.ALPHA else run_online(pl.identifier(c, alpha=a), yte, ute, y0)
            res["filter"]["sweep"].append({          # svr: decisions as taken; svr_final: after restoring
                "alpha": a, "svr": pl.precision_recall(o["flagged_rt"], an),
                "svr_final": pl.precision_recall(o["flagged"], an),
                **{f"median_{name}": pl.precision_recall(dev[name] > c["thr_median"][name][a], an)
                   for name in dev}})
    if figs:
        figures.filter_fig(tte, yte, full, OUT / "fig1_filter.png")
        figures.free_run_fig(tte, ref, sims, res["reference"], OUT / "fig2_free_run.png")
    # supervisor in ordinary operation (test data, input inside the range): how often it is not normal,
    # and before how many input changes it warns at the first opportunity (H - 2 samples ahead)
    reg, H = full["regime"], pl.HORIZON
    changes = [j for j in range(LAG + H, len(ute)) if ute[j] != ute[j - 1]]
    first = [j for j in changes if reg[first_opportunity(j, H)] > 0 and reg[first_opportunity(j, H) - 1] == 0]
    res["supervisor_test"] = {"share_not_normal": float(np.mean(reg[LAG:] > 0)),
                              "n_input_changes": len(changes), "n_warned_first_opportunity": len(first)}
    res["checks"] = checks.run(c["Z"], c["tz"], c["sigma"], c["lam"], pl.SIGMAS, pl.LAMS, seed)
    res["checks"]["gradient_physical_max_rel_dev"] = checks.gradient_physical(models["Kernel ridge"], c["Xc"], seed)
    return res, c


def drift(seed, n0=2500, n1=3000, k0_end=0.8, figs=True):
    """Catalyst slowly loses activity: the rate constant falls linearly to k0_end.
    Two measures: one-step error, and free-run of each quarter by the model frozen at the
    start of that quarter (what a controller would rely on until the next update)."""
    rng, noise_seed = data.streams(seed, SCENARIO["drift"])
    Tc = data.aprbs(n0 + n1, *TC_RANGE, 10, 60, rng)
    scale = np.r_[np.ones(n0), np.linspace(1.0, k0_end, n1)]
    df = data.make_log(Tc, noise_seed, k0_scale=scale)
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
            "filter": pl.precision_recall(o["flagged_rt"], an) if strategies[name]["use_filter"] else None,
            "filter_final": pl.precision_recall(o["flagged"], an) if strategies[name]["use_filter"] else None}
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
    rng, noise_seed = data.streams(seed, SCENARIO["excursion"])
    parts = list(zip(EXC_PARTS, (TC_RANGE, (284.0, 290.0), TC_RANGE)))
    Tc = np.concatenate([data.aprbs(n, lo, hi, 10, 60, rng) for n, (lo, hi) in parts])
    df = data.make_log(Tc, noise_seed)
    y, u, true = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true"))
    y0 = pl.first_values(y, c["y_clean"][-1])
    runs = {"Static model": run_online(pl.identifier(c, adapt=False, horizon=pl.HORIZON), y, u, y0,
                                       probe_u=PROBE_TC),
            "Sliding window": run_online(pl.identifier(c, adapt=True, horizon=pl.HORIZON), y, u, y0,
                                         probe_u=PROBE_TC)}
    n_in, n_out, n_back = EXC_PARTS
    inside = np.r_[np.ones(n_in, bool), np.zeros(n_out, bool), np.ones(n_back, bool)]
    inside[:LAG] = False
    outside = ~inside
    outside[:LAG] = False
    res = {"outside_range": [284.0, 290.0], "th1": c["lam"], "th2": 3 * c["lam"], "horizon": pl.HORIZON,
           "max_lead": n_in - first_opportunity(n_in, pl.HORIZON),   # an input change is first seen this early
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
            "point_only": _regime_stats(reg_point, n_in, n_in + n_out, pl.HORIZON),
            "horizon": _regime_stats(o["regime"], n_in, n_in + n_out, pl.HORIZON),
            "mean_penalty_multiplier_outside": float(np.nanmean(o["mult"][outside])),
            "one_step_rmse_inside": pl.rmse(o["pred"], true, inside),
            "one_step_rmse_outside": pl.rmse(o["pred"], true, outside),
            "filter_outside": pl.precision_recall(o["flagged_rt"] & outside, an & outside),
            "jump_samples_outside": int(o["jump"][outside].sum())}
    if figs:
        figures.supervisor_fig(np.arange(len(u)) * data.TS, u, runs, c["lam"], TC_RANGE,
                               OUT / "fig4_supervisor.png")
    return res


def hot_excursion(c, seed, figs=True, cold_limit=None):
    """The input exceeds the upper edge of the range by 1 K for HOT_LEN samples, past the
    point where the cold steady state disappears, and comes back."""
    rng, noise_seed = data.streams(seed, SCENARIO["hot"])
    Tc = np.r_[data.aprbs(HOT_START, *TC_RANGE, 10, 60, rng), np.full(HOT_LEN, HOT_TC),
               data.aprbs(HOT_TOTAL - HOT_START - HOT_LEN, *TC_RANGE, 10, 60, rng)]
    df = data.make_log(Tc, noise_seed)
    y, u, true, T = (df[k].to_numpy() for k in ("Ca_mol_L", "Tc_K", "Ca_true", "T_true"))
    y0 = pl.first_values(y, c["y_clean"][-1])
    runs = {"Static model": run_online(pl.identifier(c, adapt=False, horizon=pl.HORIZON), y, u, y0),
            "Sliding window": run_online(pl.identifier(c, adapt=True, horizon=pl.HORIZON), y, u, y0)}
    hot = slice(HOT_START, HOT_START + HOT_LEN)
    ca_lo = float(np.min(c["y_clean"]))
    below = np.where(true[HOT_START:] < ca_lo)[0] + HOT_START      # from the step on, not before it
    below_min = np.where(true[HOT_START:] < CA_MIN)[0] + HOT_START
    k0 = first_opportunity(HOT_START, pl.HORIZON)   # first sample whose plan contains the hot input
    res = {"tc_hot": HOT_TC, "n_hot": HOT_LEN, "T_max": float(T.max()), "Ca_min": float(true.min()),
           "T_before": float(T[HOT_START - 1]), "Ca_end": float(true[-1]), "T_end": float(T[-1]),
           "ca_min_startup": ca_lo, "first_opportunity": k0, "hot_start": HOT_START,
           "first_sample_below_startup_ca": int(below[0]) if len(below) else None,
           "first_sample_below_ca_min": int(below_min[0]) if len(below_min) else None,
           "runs": {}}
    for name, o in runs.items():
        reg = o["regime"]
        nonnormal = np.where(reg[k0:] > 0)[0]
        conservative = np.where(reg[k0:] == 2)[0]
        res["runs"][name] = {
            "normal_before": bool(reg[k0 - 1] == 0),   # otherwise a warning cannot be attributed to 304 K
            "first_non_normal": int(nonnormal[0] + k0) if len(nonnormal) else None,
            "first_conservative": int(conservative[0] + k0) if len(conservative) else None,
            "var_h_before_ignition": float(np.nanmax(o["var_h"][k0:HOT_START])),
            "one_step_rmse_last_200": pl.rmse(o["pred"][-200:], true[-200:])}
        # measurements replaced by the prediction in real time and restored later, during the hot step
        hid = np.where(o["flagged_rt"][hot] & ~o["flagged"][hot])[0] + hot.start
        res["runs"][name]["hidden_in_real_time"] = [int(k) for k in hid]
        res["runs"][name]["hidden_max_overstatement"] = (float(np.max(o["y_rt"][hid] - y[hid])) if len(hid) else None)
    if figs:
        figures.hot_fig(np.arange(len(u)) * data.TS, u, true, T, runs, c["lam"], TC_RANGE,
                        cold_limit, OUT / "fig5_hot_side.png")
    return res


def _ignition(p, tc_start, tc_step, delay):
    """Open-loop test of a lower bound on Ca as an ignition guard. From the cold steady
    state at tc_start the input steps to tc_step; `delay` min after the true Ca falls to
    CA_MIN the input drops to the strongest cooling in the range, TC_RANGE[0] (the most a
    controller could do, at once). Returns the time of the crossing and the peak T."""
    from scipy.integrate import solve_ivp
    x0 = cstr.steady_states(tc_start, p)[0]
    hit = lambda t, x, *a: x[0] - CA_MIN
    hit.terminal, hit.direction = True, -1
    kw = dict(rtol=1e-9, atol=1e-11, max_step=0.01)
    s1 = solve_ivp(cstr.rhs, (0.0, 60.0), x0, args=(tc_step, p), events=hit, **kw)
    if not s1.t_events[0].size:
        return {"t_cross_min": None, "T_max": float(s1.y[1].max())}
    T_max, x = s1.y[1].max(), s1.y_events[0][0]
    if delay > 0:
        s2 = solve_ivp(cstr.rhs, (0.0, delay), x, args=(tc_step, p), **kw)
        T_max, x = max(T_max, s2.y[1].max()), s2.y[:, -1]
    s3 = solve_ivp(cstr.rhs, (0.0, 60.0), x, args=(TC_RANGE[0], p), **kw)
    return {"t_cross_min": float(s1.t_events[0][0]), "T_max": float(max(T_max, s3.y[1].max()))}


def process_facts():
    """Steady-state facts about the simulated reactor across the identified range, the
    stability limit under parameter changes, and a dynamic test of the Ca lower bound."""
    from scipy.optimize import brentq
    rows = []
    for Tc_ in (TC_RANGE[0], 297.0, 300.0, 302.0, TC_RANGE[1]):
        ss = cstr.steady_states(Tc_)
        ca0, T0 = ss[0]
        gain = (cstr.steady_states(Tc_ + 0.01)[0][0] - ca0) / 0.01
        eig = cstr.jacobian_eigenvalues([ca0, T0], Tc_)
        rows.append({"Tc": Tc_, "Ca": float(ca0), "T": float(T0), "gain": float(gain),
                     "slowest_time_constant_min": float(1.0 / np.min(-eig.real)),
                     "n_steady_states": len(ss)})
    sens = []
    for name, ch in SENSITIVITY.items():
        lim, ca, T = cstr.saddle_node(params(ch))
        fold = bool(np.isfinite(lim))      # False: a single steady state at every Tc, nothing to ignite
        sens.append({"case": name, "fold": fold, "tc_limit": lim if fold else None,
                     "ca_at_limit": ca if fold else None, "T_at_limit": T if fold else None,
                     "ca_min_excludes_limit": bool(CA_MIN > ca) if fold else None})
    # feed-concentration increase at which Ca at the limit reaches CA_MIN
    caf_rise = {name: float(brentq(lambda m: cstr.saddle_node(params({**ch, "Caf": m}))[1] - CA_MIN, 1.0, 1.3)
                            - 1.0)
                for name, ch in (("nominal", {}), ("UA -10 %", {"UA": 0.90}))}
    ign = []
    for name, ch in IGNITION_CASES.items():
        p = params(ch)
        lim = cstr.saddle_node(p)[0]
        start = round(lim - IGNITION_MARGIN_K, 2)
        ign.append({"case": name, "tc_limit": lim, "tc_start": start, "tc_step": TC_RANGE[1],
                    "ca_start": float(cstr.steady_states(start, p)[0][0]),
                    "runs": [{"delay_min": d, **_ignition(p, start, TC_RANGE[1], d)} for d in DELAYS_MIN]})
    hot = []
    for ca, T in cstr.steady_states(HOT_TC):
        eig = cstr.jacobian_eigenvalues([ca, T], HOT_TC)
        hot.append({"Ca": float(ca), "T": float(T), "max_real_eig": float(np.max(eig.real)),
                    "eig": [str(np.round(e, 3)) for e in eig]})
    return {"cold_branch_limit_K": sens[0]["tc_limit"], "steady_states": rows, "limit_sensitivity": sens,
            "ca_min_constraint": CA_MIN, "caf_rise_breaking_ca_min": caf_rise, "ignition_delay": ign,
            "ignition_margin_K": IGNITION_MARGIN_K, "cool_tc": TC_RANGE[0],
            "steady_states_at_hot_tc": hot, "saddle_node_check": checks.saddle_node()}


def simulated_run(seed, figs=True, cold_limit=None):
    rng, noise_seed = data.streams(seed, SCENARIO["stationary"])
    df = data.make_log(data.aprbs(8000, *TC_RANGE, 10, 60, rng), noise_seed, gap_at=2500)
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
                "n_jump": N_JUMP, "refit_every_jump": REFIT_EVERY_JUMP, "lags": [NA, NB], "horizon": pl.HORIZON,
                "tc_range": list(TC_RANGE), "median_window": data.MEDIAN_WINDOW, "loo_tol": pl.LOO_TOL,
                "holdout": pl.HOLDOUT, "fr_tol": pl.FR_TOL, "hot_tc": HOT_TC, "anomaly_rate": data.ANOMALY_RATE,
                "startup_share": STARTUP_SHARE, "min_cal_rows": pl.MIN_CAL_ROWS}
    # one BLAS thread, as in run_seeds.py: the reference run and the seed spread are computed the same
    # way (with more threads the summation order changes and the filter threshold moves in the 3rd digit)
    with threadpool_limits(1):
        if args.data:
            df = pd.read_csv(args.data)
            ts = data.validate_log(df, MIN_LOG)
            n_tr = int(STARTUP_SHARE * len(df))
            pl.startup_check(df["Ca_mol_L"].to_numpy()[:n_tr], df["Tc_K"].to_numpy()[:n_tr], STARTUP_SHARE)
            summary = {"simulated": False, "sampling_min": ts, "settings": settings, "data": data.describe(df)}
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
