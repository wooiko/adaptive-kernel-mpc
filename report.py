"""Fills the validation report from the computed numbers (no hand-typed results)."""
import numpy as np


def _pr(p):
    return (f"{p['precision']:.2f} / {p['recall']:.2f} ({p['caught']} of {p['true_anomalies']} caught, "
            f"{p['flagged']} flagged; {100 * p['false_rejection_share']:.2f} % of normal samples rejected)")


def _lead(x):
    return "already warning" if x is None else str(x)


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _spread(values, nd=4, pct=False):
    v = np.array([x for x in values if x is not None], float)
    if len(v) == 0:
        return "n/a"
    q = np.percentile(v, [50, 25, 75])
    if pct:
        return f"{100 * q[0]:.1f} % [{100 * q[1]:.1f}; {100 * q[2]:.1f}]"
    return f"{q[0]:.{nd}f} [{q[1]:.{nd}f}; {q[2]:.{nd}f}]"


SEED_METRICS = [
    # key, label, decimals, percent
    ("noise_ratio", "Estimated / true sensor noise", 3, False),
    ("krr_free_run", "Free-run RMSE, kernel model", 4, False),
    ("arx_free_run", "Free-run RMSE, ARX", 4, False),
    ("oe_free_run", "Free-run RMSE, OE", 4, False),
    ("arx_over_krr", "Free-run RMSE ratio ARX / kernel", 2, False),
    ("oe_over_krr", "Free-run RMSE ratio OE / kernel", 2, False),
    ("svr_precision", "SVR filter precision (default share)", 2, False),
    ("svr_recall", "SVR filter recall (default share)", 2, False),
    ("svr_false_share", "SVR filter: normal samples rejected", 2, True),
    ("median_precision", "Causal median precision (same share)", 2, False),
    ("median_recall", "Causal median recall (same share)", 2, False),
    ("filter_gain", "One-step RMSE ratio without / with filter", 2, False),
    ("drift_one_step_ratio", "Drift, one-step: static no filter / full system", 2, False),
    ("drift_filter_effect", "Drift, one-step: static no filter / static + filter", 2, False),
    ("drift_adapt_effect_q4", "Drift, free-run Q4: static + filter / full system", 2, False),
    ("exc_window_nonnormal", "Cold excursion, window: time outside not normal", 1, True),
    ("exc_window_lead_return", "Cold excursion, window: warning before return, samples", 0, False),
    ("exc_window_back_to_normal", "Cold excursion, window: samples to normal after leaving", 0, False),
    ("exc_static_nonnormal", "Cold excursion, static: time outside not normal", 1, True),
    ("hot_var_before_over_th1", "Hot side: largest planned-trajectory variance before ignition / th1", 2, False),
    ("hot_warning_lag", "Hot side: first warning relative to Ca leaving the start-up range, samples "
                        "(negative = before)", 0, False),
]


def write_report(s, path, seeds=None):
    d, st, cfg = s["data"], s["stationary"], s["settings"]
    names = list(st["models"])
    krr = st["models"]["Kernel ridge"]
    L = ["# Adaptive kernel model of a CSTR: validation report", "",
         "Process: continuous stirred-tank reactor, exothermic reaction A -> B. "
         "Input: coolant temperature Tc. Output: concentration Ca. All errors are in mol/L.",
         ("Data: simulated experiments on the textbook model (Henson & Seborg, 1997) with sensor noise, "
          "isolated spikes in 2 % of the samples and a logger gap added on purpose."
          if s["simulated"] else "Data: supplied log file."),
         (f"Sections 1-7 are one realisation (seed {s.get('seed', 0)}); section 8 repeats the experiments "
          "over many seeds." if s["simulated"] and seeds else ""), "",
         "The identification system has three levels that share one RBF kernel:", "",
         "1. an SVR filter rejects anomalous measurements;",
         f"2. kernel ridge regression (KRR) on a sliding window of {cfg['window']} samples is the process model;",
         "3. a Gaussian-process (GP) supervisor measures how far the trajectory planned from the current "
         "operating point is from the training data.",
         "", "## 1. Data", "",
         f"- Samples: {d['n_samples']}, sampling period {s['sampling_min']} min; missing values: {d['n_missing']}.",
         f"- Estimated sensor noise: {st['noise_std_est']:.5f} (standard deviation; second differences, "
         "robust scale with outlier trimming).",
         f"- Input range: {d['input_min']:.1f} to {d['input_max']:.1f} K, {d['input_levels']} distinct levels; "
         f"the least-visited tenth of the range holds {100 * d['input_emptiest_decile_share']:.1f} % of samples.",
         f"- Start-up (cleaning, tuning, threshold calibration, first model): first {st['n_startup']} samples. "
         f"Test: last {st['n_test']} samples, never used for fitting, tuning or calibration."]
    f = st["filter"]
    L += ["", "## 2. Level 1: anomaly filter", "",
          "A measurement is rejected when it differs from the SVR one-step prediction by more than a threshold. "
          "The threshold is calibrated on start-up data so that a chosen share of normal samples is rejected: "
          f"the filter is replayed as online (an SVR on the last {cfg['window']} rows, refitted every "
          f"{cfg['refit_every']} rows, predicts the rows that follow), and the threshold is the corresponding quantile "
          "of these out-of-sample residuals, divided by the in-sample robust scale of the SVR that produced them "
          f"({f['n_calibration_rows']} rows). The residuals have heavier tails than a normal "
          "distribution, so the threshold is not a number of Gaussian standard deviations.",
          f"Default share: {100 * f['alpha']:.1f} %; threshold {f['threshold'][str(f['alpha'])]:.2f} "
          f"times the in-sample robust scale. On the test data the filter rejected {f['n_flagged_test']} samples."]
    if "sweep" in f:
        L += ["", "Precision / recall against the known injected spikes, test data. Every row is the full adaptive "
              "system. The rolling medians (window 7) are calibrated to the same share on the same start-up data.", "",
              "| Target share of normal samples rejected | SVR filter | Rolling median, causal | "
              "Rolling median, centred (non-causal: needs 3 future samples) |", "|---|---|---|---|"]
        L += [f"| {100 * r['alpha']:.1f} % | {_pr(r['svr'])} | {_pr(r['median_causal'])} | {_pr(r['median_centred'])} |"
              for r in f["sweep"]]
        dflt = next(r for r in f["sweep"] if r["alpha"] == f["alpha"])
        a, b = dflt["svr"], dflt["median_causal"]
        L += ["", f"At the default share the SVR filter has precision {a['precision']:.2f} and recall {a['recall']:.2f}; "
              f"the causal rolling median, the detector usable online without delay, has {b['precision']:.2f} and "
              f"{b['recall']:.2f}. The centred median sees 3 future samples and could only run with a 0.3-min delay.",
              f"With the filter the one-step error of the window model is {f['one_step_rmse_with_filter']:.4f}; "
              f"without it, {f['one_step_rmse_without_filter']:.4f} "
              f"({f['one_step_rmse_without_filter'] / f['one_step_rmse_with_filter']:.1f} times larger)."]
    L += ["", "![filter](fig1_filter.png)", "",
          "## 3. Level 2: process model", "",
          "KRR and two linear references predict the next sample of Ca from the last two samples of Ca and Tc: "
          "ARX (least squares on the one-step error) and OE (least squares on the free-run error). "
          f"Kernel width {st['sigma']:.2f} and regularisation {st['lam']:.4f} (normalised units) were chosen by exact "
          "leave-one-out cross-validation.",
          f"Errors are measured against the {st['reference']}. "
          "Free-run: the model is fed its own predictions, as inside a controller.", "",
          "| Model | One-step RMSE | Free-run RMSE | Free-run max error | Free-run fit |", "|---|---|---|---|---|"]
    for name, m in st["models"].items():
        L.append(f"| {name} | {m['one_step_rmse']:.4f} | {m['free_run_rmse']:.4f} | "
                 f"{m['free_run_max_abs_err']:.4f} | {m['free_run_fit_percent']:.1f} % |")
    lin = [n for n in names if n != "Kernel ridge"]
    L += ["", "The kernel model's free-run error is " + " and ".join(
              f"{st['models'][n]['free_run_rmse'] / krr['free_run_rmse']:.1f} times smaller than {n.split(' ')[0]}'s"
              for n in lin) + ".",
          "ARX coefficients are biased by the noise in the lagged outputs; OE removes that bias, so the comparison "
          "with OE is the one that isolates the effect of the nonlinearity.",
          f"The error is not uniform: where Ca is in its lowest tenth (below {st['ca_low_threshold']:.3f}, the hot end, "
          f"where the reactor gain is steepest) the kernel model's free-run RMSE is {st['krr_free_run_rmse_low_ca']:.4f}; "
          f"elsewhere it is {st['krr_free_run_rmse_elsewhere']:.4f}.",
          "", "![free run](fig2_free_run.png)", ""]
    if "process" in s:
        pr = s["process"]
        rows = pr["steady_states"]
        L += ["## 4. Operating range and stability of the reactor", "",
              "Steady states of the simulated reactor on the cold (low-temperature) branch, computed from the model "
              "equations:", "",
              "| Tc, K | Ca, mol/L | Gain dCa/dTc, mol/(L K) | Slowest time constant, min | Steady states |",
              "|---|---|---|---|---|"]
        L += [f"| {r['Tc']:.1f} | {r['Ca']:.4f} | {r['gain']:.4f} | {r['slowest_time_constant_min']:.2f} | "
              f"{r['n_steady_states']} |" for r in rows]
        lim, top = pr["cold_branch_limit_K"], rows[-1]
        L += ["", f"Across the identified range the gain grows {rows[-1]['gain'] / rows[0]['gain']:.0f}-fold and the "
              f"dynamics at the upper edge are {top['slowest_time_constant_min'] / min(r['slowest_time_constant_min'] for r in rows):.1f} "
              f"times slower than at their fastest. In the upper part of the range three steady states coexist. "
              f"The cold steady state disappears at Tc = {lim:.2f} K (saddle-node point), {lim - top['Tc']:.2f} K above "
              "the upper edge of the identified range; above it only the hot steady state remains, and it is unstable "
              "(eigenvalues with positive real part), so the reactor ignites and cannot settle (section 7).",
              "", "The saddle-node point moves strongly with the process parameters, while Ca at that point moves little:",
              "", "| Parameters | Tc at the limit, K | Ca at the limit, mol/L | T at the limit, K |", "|---|---|---|---|"]
        L += [f"| {r['case']} | {r['tc_limit']:.2f} | {r['ca_at_limit']:.3f} | {r['T_at_limit']:.1f} |"
              for r in pr["limit_sensitivity"]]
        noise = st["noise_std_est"]
        L += ["", "A fixed margin on Tc is therefore not reliable: with 5 % less heat transfer the limit already lies "
              "inside the identified range. Constraints for a controller:",
              f"- Tc <= {top['Tc']:.1f} K, the edge of the data (no extrapolation of the model);",
              f"- Ca >= {pr['ca_min_constraint']:.2f} mol/L, the steady-state Ca at the edge of the data rounded up. "
              f"It stays above Ca at the limit in every case of the table, by at least {pr['ca_margin_to_worst_limit']:.3f} "
              f"mol/L ({pr['ca_margin_to_worst_limit'] / noise:.0f} times the sensor noise).",
              "Closed-loop behaviour with these constraints is not tested: there is no controller in this part.", ""]
    if "drift" in s:
        dr = s["drift"]
        S = dr["strategies"]
        L += ["## 5. Slow drift of the process", "",
              f"Catalyst activity falls linearly to {100 * dr['k0_end']:.0f} % over {dr['n_online']} samples. "
              "The steady-state Ca at the end is higher by " + ", ".join(
                  f"{v:.3f} at {float(k):.0f} K" for k, v in dr["ca_shift_at_k0_end"].items()) + ".",
              "", "One-step RMSE against the true output, by quarter of the run:", "",
              "| Strategy | Q1 | Q2 | Q3 | Q4 | Whole run |", "|---|---|---|---|---|---|"]
        for name, r in S.items():
            L.append(f"| {name} | " + " | ".join(f"{q:.4f}" for q in r["rmse_by_quarter"]) + f" | {r['rmse']:.4f} |")
        L += ["", "The one-step error is dominated by the measurement history and hardly grows with the drift. "
              "Without adaptation the model is the same with or without the filter, so the two static rows of the "
              "next table coincide. "
              "The test that matters for a controller is the free run: each quarter is simulated by the model "
              "frozen at the start of that quarter. RMSE / largest error:", "",
              "| Strategy | Q1 | Q2 | Q3 | Q4 |", "|---|---|---|---|---|"]
        for name, r in S.items():
            L.append(f"| {name} | " + " | ".join(f"{q['rmse']:.4f} / {q['max_abs_err']:.4f}"
                                              for q in r["free_run_by_quarter"]) + " |")
        a0, a1 = S["Static model, no filter"], S["Static model + SVR filter"]
        w1 = S["SVR filter + sliding window"]
        L += ["", "Contributions, separated:",
              f"- filter (both models static), one-step RMSE: {a0['rmse']:.4f} without, {a1['rmse']:.4f} with;",
              f"- adaptation (both with the filter), free-run RMSE in Q4: {a1['free_run_by_quarter'][3]['rmse']:.4f} static, "
              f"{w1['free_run_by_quarter'][3]['rmse']:.4f} with the sliding window "
              f"({a1['free_run_by_quarter'][3]['rmse'] / w1['free_run_by_quarter'][3]['rmse']:.1f} times);",
              f"- a frozen filter on a drifting process rejects valid data: static filter {_pr(a1['filter'])}; "
              f"filter refitted with the window {_pr(w1['filter'])}.",
              "", "![drift](fig3_drift.png)", ""]
    if "excursion" in s:
        ex = s["excursion"]
        a, b = ex["runs"]["Static model"], ex["runs"]["Sliding window"]
        lo, hi = ex["outside_range"]
        H = ex["horizon"]
        pt = lambda r: f"{100 * sum(r[1:]):.0f} %"
        L += ["## 6. Level 3: leaving the training range (cold side)", "",
              f"The coolant temperature moves to {lo:.0f}-{hi:.0f} K, never seen during start-up, and returns. "
              f"The supervisor receives the largest GP variance along the free-run of the model over the next {H} "
              f"samples ({H * s['sampling_min']:.0f} min) under the planned input; in this open-loop test the plan is "
              "the input schedule itself, in a predictive controller it is the candidate input sequence. "
              f"It is cautious above {ex['th1']:.4f} and conservative above {ex['th2']:.4f}; the penalty on control moves "
              "rises continuously up to twofold. The thresholds are a heuristic: the GP prior variance is fixed at 1, "
              "so the variance measures data coverage, not a calibrated error.", "",
              "| | Static model | Sliding window |", "|---|---|---|",
              f"| GP variance inside the range, current point (median) | {a['var_median_inside']:.5f} | {b['var_median_inside']:.5f} |",
              f"| GP variance outside, current point (median) | {a['var_median_outside']:.5f} | {b['var_median_outside']:.5f} |",
              f"| GP variance outside, along the planned trajectory (median) | {a['var_h_median_outside']:.5f} | {b['var_h_median_outside']:.5f} |"]
        for i, tc in enumerate(ex["probe_tc"]):
            L.append(f"| GP variance outside with the input set to {tc:.1f} K (median) | "
                     f"{a['probe_var_median_outside'][i]:.5f} | {b['probe_var_median_outside'][i]:.5f} |")
        L += [f"| Time outside not normal: current point only | {pt(a['point_only_regime_share_outside'])} | "
              f"{pt(b['point_only_regime_share_outside'])} |",
              f"| Time outside not normal: planned trajectory (supervisor) | {pt(a['regime_share_outside'])} | "
              f"{pt(b['regime_share_outside'])} |",
              f"| Time inside not normal (supervisor) | {100 * sum(a['regime_share_inside'][1:]):.1f} % | "
              f"{100 * sum(b['regime_share_inside'][1:]):.1f} % |",
              f"| Warning before the input leaves the range, samples: current point / trajectory | "
              f"{_lead(a['point_only']['lead_before_leaving'])} / {_lead(a['horizon']['lead_before_leaving'])} | "
              f"{_lead(b['point_only']['lead_before_leaving'])} / {_lead(b['horizon']['lead_before_leaving'])} |",
              f"| Warning before the input returns, samples: current point / trajectory | "
              f"{_lead(a['point_only']['lead_before_return'])} / {_lead(a['horizon']['lead_before_return'])} | "
              f"{_lead(b['point_only']['lead_before_return'])} / {_lead(b['horizon']['lead_before_return'])} |",
              f"| Samples from leaving the range to the first normal sample (supervisor) | "
              f"{a['horizon']['samples_to_normal_after_leaving'] if a['horizon']['samples_to_normal_after_leaving'] is not None else 'never'} | "
              f"{b['horizon']['samples_to_normal_after_leaving'] if b['horizon']['samples_to_normal_after_leaving'] is not None else 'never'} |",
              f"| One-step RMSE inside | {a['one_step_rmse_inside']:.4f} | {b['one_step_rmse_inside']:.4f} |",
              f"| One-step RMSE outside | {a['one_step_rmse_outside']:.4f} | {b['one_step_rmse_outside']:.4f} |",
              f"| Filter outside: precision / recall | {a['filter_outside']['precision']:.2f} / {a['filter_outside']['recall']:.2f} | "
              f"{b['filter_outside']['precision']:.2f} / {b['filter_outside']['recall']:.2f} |",
              f"| Samples outside in filter jump mode | {a['jump_samples_outside']} | {b['jump_samples_outside']} |",
              "", "A frozen model stays uncertain for as long as the process is outside the range. The sliding window "
              "learns the neighbourhood of the current operating point within a few samples, so the variance at the "
              "current point alone soon reads normal. Normal then means only that the planned trajectory runs through "
              "data the window has seen; the probed variances show the model's coverage of other inputs. "
              "Because the supervisor checks the planned trajectory, it warns before the input leaves the range and "
              "before it returns, by up to the horizon length.",
              "With the window frozen (static model) the filter cannot agree with the new data, so it stays in jump "
              "mode, where single spikes are caught by the model-free trend test.",
              "", "![supervisor](fig4_supervisor.png)", ""]
    if "hot_excursion" in s:
        hx = s["hot_excursion"]
        a, b = hx["runs"]["Static model"], hx["runs"]["Sliding window"]
        th1 = s["excursion"]["th1"] if "excursion" in s else None
        L += ["## 7. Hot side: beyond the stability limit", "",
              f"The coolant temperature is held at {hx['tc_hot']:.1f} K, 1 K above the identified range and beyond the "
              f"saddle-node point, for {hx['n_hot']} samples ({hx['n_hot'] * s['sampling_min']:.0f} min), then returns.",
              f"- The reactor ignites: Ca leaves the start-up range (below {hx['ca_min_startup']:.3f}) at sample "
              f"{hx['first_sample_below_startup_ca']}, {hx['first_sample_below_startup_ca'] - 300} samples after the step; "
              f"Ca falls to {hx['Ca_min']:.3f} and the reactor temperature rises from {hx['T_before']:.1f} to "
              f"{hx['T_max']:.1f} K. After the return it settles back on the cold branch (Ca {hx['Ca_end']:.3f}, "
              f"T {hx['T_end']:.1f} K).",
              (f"- Before ignition the largest variance along the planned trajectory, which already contains "
               f"{hx['tc_hot']:.0f} K, is {a['var_h_before_ignition']:.4f} (static) and {b['var_h_before_ignition']:.4f} "
               f"(window), against the cautious threshold {_f(th1)}. The variance measures the distance from the data, "
               "not the closeness of the stability limit, and 1 K beyond the data that distance is small."),
              (f"- The first warning comes at sample {a['first_non_normal'] if a['first_non_normal'] is not None else 'none'} (static) and "
               f"{b['first_non_normal'] if b['first_non_normal'] is not None else 'none'} (window); Ca leaves the start-up range at sample "
               f"{hx['first_sample_below_startup_ca']}. A warning only raises the move penalty; it does not stop the input."),
              "", "The supervisor is therefore not a substitute for a hard input constraint (section 4); section 8 shows "
              "how often it warns in time.",
              "", "![hot side](fig5_hot_side.png)", ""]
    if seeds:
        L += [f"## 8. Spread over {seeds['n_seeds']} seeds", "",
              f"All simulated experiments repeated with seeds 0-{seeds['n_seeds'] - 1} (new input sequences, noise, "
              "spikes and start-up for each seed). Median [25th; 75th percentile]:", "",
              "| Quantity | Median [IQR] |", "|---|---|"]
        for key, label, nd, pct in SEED_METRICS:
            L.append(f"| {label} | {_spread([r.get(key) for r in seeds['per_seed']], nd, pct)} |")
        n = seeds["n_seeds"]
        lead_min = 10                     # samples = 1 min
        warned = sum(r["hot_warning_lag"] is not None and r["hot_warning_lag"] <= -lead_min for r in seeds["per_seed"])
        oe_better = sum(r["oe_over_krr"] < 1 for r in seeds["per_seed"])
        L += ["", f"Hot side: the supervisor warned at least {lead_min} samples ({lead_min * s['sampling_min']:.0f} min) "
              f"before Ca left the start-up range in {warned} of {n} seeds; even when it warns, it only raises the "
              "move penalty.",
              f"The linear OE model was more accurate than the kernel model in {oe_better} of {n} seeds.", ""]
    c = st["checks"]
    L += [f"## {9 if seeds else 8}. Numerical self-checks", "",
          f"- Analytical gradient against central differences: largest relative deviation {c['gradient_max_rel_dev']:.1e} "
          f"(normalised units), {c['gradient_physical_max_rel_dev']:.1e} (physical units, as needed for MPC).",
          f"- Closed-form leave-one-out residuals against actual refits: largest deviation {c['loo_max_abs_dev']:.1e}.",
          f"- GP variance from the KRR factorisation against a library GP: largest deviation {c['variance_max_abs_dev']:.1e}.",
          f"- Leave-one-out search over {c['loo_grid_size']} settings on {c['loo_grid_n']} samples: {c['loo_grid_seconds']:.1f} s.",
          "- These checks confirm that the code matches the formulas; they do not show that the GP variance is a "
          "calibrated error estimate.",
          "", f"## {10 if seeds else 9}. Design choices for a dynamic process", "",
          f"- The model is dynamic (NARX with {cfg['lags'][0]} past outputs and {cfg['lags'][1]} past inputs), not a static input-output map.",
          f"- Anomaly threshold: calibrated to reject {100 * cfg['alpha']:.1f} % of normal samples on start-up data "
          "(out-of-sample residual quantile).",
          f"- Window of {cfg['window']} samples, refitted every {cfg['refit_every']} accepted samples; the SVR filter is refitted together with it.",
          "- A run of 4 rejections is taken as a process change: the rejected samples are restored to the history "
          "and the window; until the filter agrees again, single spikes are caught by a model-free trend test and "
          "the models are refitted every 5 accepted samples.",
          f"- The supervisor checks the trajectory planned over {cfg['horizon']} samples, not only the current point.",
          "- The leave-one-out surface is flat near its minimum; among settings within 2 % of the best, the strongest "
          "regularisation and then the narrowest kernel are taken.",
          "- A rejected or missing sample is replaced by the model prediction in the regressor history.",
          "", f"## {11 if seeds else 10}. Limits", "",
          f"- Validated for Tc between {d['input_min']:.1f} and {d['input_max']:.1f} K at a sampling period of {s['sampling_min']} min.",
          "- Simulation study on a textbook process; no plant data.",
          "- The supervisor's penalty multiplier is computed but not yet used: there is no controller in this part.",
          "- Anomalies are isolated spikes only; a sensor fault that lasts several samples is treated as a process change.",
          ""]
    path.write_text("\n".join(L))
