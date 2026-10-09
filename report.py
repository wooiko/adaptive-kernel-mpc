"""Fills the validation report from the computed numbers (no hand-typed results)."""
import numpy as np


def _pr(p):
    return (f"{p['precision']:.2f} / {p['recall']:.2f} ({p['caught']} of {p['true_anomalies']} caught, "
            f"{p['flagged']} flagged; {100 * p['false_rejection_share']:.2f} % of normal samples rejected)")


def _lead(x):
    return "already warning" if x is None else str(x)


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _n(x):
    return "none" if x is None else str(x)


def _seeds(k):
    return f"{k} seed" + ("" if k == 1 else "s")


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
    ("tune_chosen_free_run", "Tuning: held-out free-run RMSE of the chosen setting", 4, False),
    ("tune_loo_min_free_run", "Tuning: held-out free-run RMSE of the LOO-minimum setting", 4, False),
    ("tune_worst_candidate_free_run", "Tuning: worst held-out free-run RMSE among LOO candidates", 4, False),
    ("tune_chosen_over_best", "Tuning: held-out free run, chosen / best candidate", 2, False),
    ("lam_over_noise_var", "Selected lam / sensor-noise variance (normalised units)", 1, False),
    ("svr_precision", "SVR filter precision, real time (default share)", 2, False),
    ("svr_recall", "SVR filter recall, real time (default share)", 2, False),
    ("svr_false_share", "SVR filter: normal samples rejected, real time", 2, True),
    ("svr_false_share_final", "SVR filter: normal samples rejected, after restores", 2, True),
    ("median_precision", "Causal median precision (same share)", 2, False),
    ("median_recall", "Causal median recall (same share)", 2, False),
    ("filter_gain", "One-step RMSE ratio without / with filter", 2, False),
    ("drift_one_step_ratio", "Drift, one-step: static no filter / full system", 2, False),
    ("drift_filter_effect", "Drift, one-step: static no filter / static + filter", 2, False),
    ("drift_adapt_effect_q4", "Drift, free-run Q4: static + filter / full system", 2, False),
    ("exc_window_nonnormal", "Cold excursion, window: time outside not normal", 1, True),
    ("exc_window_inside_nonnormal", "Cold excursion, window: time inside the range not normal", 1, True),
    ("exc_window_lead_return", "Cold excursion, window: warning before return, samples", 0, False),
    ("exc_window_back_to_normal", "Cold excursion, window: samples to normal after leaving", 0, False),
    ("exc_static_nonnormal", "Cold excursion, static: time outside not normal", 1, True),
    ("hot_var_before_over_th1", "Hot side: largest planned-trajectory variance before ignition / th1", 2, False),
    ("hot_warning_lag_window", "Hot side, window: first warning relative to Ca leaving the start-up range, "
                               "samples (negative = before)", 0, False),
    ("hot_warning_lag_static", "Hot side, static: the same", 0, False),
]


def write_report(s, path, seeds=None):
    d, st, cfg = s["data"], s["stationary"], s["settings"]
    ts = s["sampling_min"]
    names = list(st["models"])
    krr = st["models"]["Kernel ridge"]
    nj = cfg["n_jump"]
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
         f"- Samples: {d['n_samples']}, sampling period {ts:g} min; missing values: {d['n_missing']}.",
         f"- Estimated sensor noise: {st['noise_std_est']:.5f} (standard deviation; second differences, "
         "robust scale with outlier trimming).",
         f"- Input range: {d['input_min']:.1f} to {d['input_max']:.1f} K, {d['input_levels']} distinct levels; "
         f"the least-visited tenth of the range holds {100 * d['input_emptiest_decile_share']:.1f} % of samples.",
         f"- Start-up (cleaning, tuning, threshold calibration, first model): first {st['n_startup']} samples. "
         f"Test: last {st['n_test']} samples, never used for fitting, tuning or calibration."]

    # ---------------------------------------------------------------- 2. filter
    f = st["filter"]
    n_rt, n_fin = f["n_flagged_test"], f["n_flagged_test_final"]
    L += ["", "## 2. Level 1: anomaly filter", "",
          "A measurement is rejected when it differs from the SVR one-step prediction by more than a threshold; "
          "the rejected value is replaced by the model prediction in the history. "
          "The threshold is calibrated on start-up data so that a chosen share of normal samples is rejected: "
          f"the filter is replayed as online (an SVR on the last {cfg['window']} rows, refitted every "
          f"{cfg['refit_every']} rows, predicts the rows that follow), and the threshold is the corresponding quantile "
          "of these out-of-sample residuals, divided by the in-sample robust scale of the SVR that produced them "
          f"({f['n_calibration_rows']} rows). The residuals have heavier tails than a normal "
          "distribution, so the threshold is not a number of Gaussian standard deviations.",
          f"A run of {nj} rejections in a row is taken as a process change, and the rejected samples of the run are "
          "put back into the history afterwards. Two counts therefore exist: decisions as taken at each sample "
          "(real time, what a controller receives) and the final flags after these restores. "
          "The real-time count is the primary one.",
          f"Default share: {100 * f['alpha']:.1f} %; threshold {f['threshold'][str(f['alpha'])]:.2f} "
          f"times the in-sample robust scale. On the test data the filter rejected {n_rt} samples in real time; "
          f"{n_rt - n_fin} of them were restored later as parts of process changes, leaving {n_fin}."]
    if "sweep" in f:
        hw = cfg["median_window"]
        fut = (hw - 1) // 2
        L += ["", "Precision / recall against the known injected spikes, test data. Every row is the full adaptive "
              f"system. The rolling medians (window {hw}) have no restores, so they compare with the real-time column; "
              "they are calibrated to the same share on the same start-up data.", "",
              "| Target share of normal samples rejected | SVR filter, real time | SVR filter, after restores | "
              f"Rolling median, causal | Rolling median, centred (non-causal: needs {fut} future samples) |",
              "|---|---|---|---|---|"]
        L += [f"| {100 * r['alpha']:.1f} % | {_pr(r['svr'])} | {_pr(r['svr_final'])} | {_pr(r['median_causal'])} | "
              f"{_pr(r['median_centred'])} |" for r in f["sweep"]]
        dflt = next(r for r in f["sweep"] if r["alpha"] == f["alpha"])
        a, b, af = dflt["svr"], dflt["median_causal"], dflt["svr_final"]
        L += ["", f"At the default share the SVR filter has precision {a['precision']:.2f} and recall {a['recall']:.2f} "
              f"in real time; the causal rolling median, the detector usable online without delay, has "
              f"{b['precision']:.2f} and {b['recall']:.2f}. The centred median sees {fut} future samples and could only "
              f"run with a {fut * ts:g}-min delay.",
              f"In real time the filter rejected {100 * a['false_rejection_share']:.2f} % of normal samples against the "
              f"calibration target of {100 * f['alpha']:.1f} % ({100 * af['false_rejection_share']:.2f} % after restores). "
              "The difference between the two counts consists of normal samples rejected in runs that were later "
              "recognised as process changes; the calibration tests single samples and does not include them.",
              f"With the filter the one-step error of the window model is {f['one_step_rmse_with_filter']:.4f}; "
              f"without it, {f['one_step_rmse_without_filter']:.4f} "
              f"({f['one_step_rmse_without_filter'] / f['one_step_rmse_with_filter']:.1f} times larger)."]
    L += ["", "![filter](fig1_filter.png)", ""]

    # ---------------------------------------------------------------- 3. model
    tu = st["tuning"]
    L += ["## 3. Level 2: process model", "",
          "KRR and two linear references predict the next sample of Ca from the last two samples of Ca and Tc: "
          "ARX (least squares on the one-step error) and OE (least squares on the free-run error).",
          f"Kernel width {st['sigma']:.2f} and regularisation {st['lam']:.4f} (normalised units) were chosen in two "
          f"stages. (1) Exact leave-one-out (LOO) RMSE over a grid, on a random subset of the first "
          f"{100 * (1 - cfg['holdout']):.0f} % of the start-up rows: {tu['n_candidates_loo']} settings lie within "
          f"{100 * cfg['loo_tol']:.0f} % of the best. LOO measures the one-step error, and these candidates differ in "
          f"free run: on the held-out last {100 * cfg['holdout']:.0f} % of the start-up block ({tu['n_holdout']} "
          f"samples) their free-run RMSE ranges from {tu['holdout_free_run_rmse_range'][0]:.4f} to "
          f"{tu['holdout_free_run_rmse_range'][1]:.4f}. (2) Settings within {100 * cfg['fr_tol']:.0f} % of the best "
          f"held-out free run are admissible ({tu['n_admissible']}); among them the strongest regularisation and then "
          "the narrowest kernel are taken: the smoothest model whose uncertainty still grows quickly away from the data. "
          f"The chosen setting: LOO {tu['chosen']['loo_over_min']:.4f} times the minimum, held-out free-run RMSE "
          f"{tu['chosen']['holdout_free_run_rmse']:.4f}; the LOO-minimum setting (width {tu['loo_minimum']['sigma']:.2f}, "
          f"regularisation {tu['loo_minimum']['lam']:.1e}) gives {tu['loo_minimum']['holdout_free_run_rmse']:.4f}.",
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
          "KRR, like ARX, is fitted on the one-step (equation) error with noisy lagged outputs in the regressor; "
          "OE is fitted on the free-run error, which removes the bias that this noise causes in the linear "
          "coefficients. KRR against ARX therefore compares nonlinear with linear under the same criterion; "
          "KRR against OE compares it with the best linear simulation model, which has the advantage of the criterion.",
          f"The error is not uniform: where Ca is in its lowest tenth (below {st['ca_low_threshold']:.3f}, the hot end, "
          f"where the reactor gain is steepest) the kernel model's free-run RMSE is {st['krr_free_run_rmse_low_ca']:.4f}; "
          f"elsewhere it is {st['krr_free_run_rmse_elsewhere']:.4f}.",
          "", "![free run](fig2_free_run.png)", ""]

    # ---------------------------------------------------------------- 4. process
    if "process" in s:
        pr = s["process"]
        rows = pr["steady_states"]
        ca_min = pr["ca_min_constraint"]
        L += ["## 4. Operating range and stability of the reactor", "",
              "Steady states of the simulated reactor on the cold (low-temperature) branch, computed from the model "
              "equations:", "",
              "| Tc, K | Ca, mol/L | Gain dCa/dTc, mol/(L K) | Slowest time constant, min | Steady states |",
              "|---|---|---|---|---|"]
        L += [f"| {r['Tc']:.1f} | {r['Ca']:.4f} | {r['gain']:.4f} | {r['slowest_time_constant_min']:.2f} | "
              f"{r['n_steady_states']} |" for r in rows]
        lim, top = pr["cold_branch_limit_K"], rows[-1]
        hot = pr["steady_states_at_hot_tc"]
        hot_txt = "; ".join(f"Ca {h['Ca']:.3f}, T {h['T']:.1f} K, largest real part of the eigenvalues "
                            f"{h['max_real_eig']:.3f} 1/min" for h in hot)
        L += ["", f"Across the identified range the gain grows {rows[-1]['gain'] / rows[0]['gain']:.0f}-fold and the "
              f"dynamics at the upper edge are {top['slowest_time_constant_min'] / min(r['slowest_time_constant_min'] for r in rows):.1f} "
              f"times slower than at their fastest. In the upper part of the range three steady states coexist. "
              f"The cold steady state disappears at Tc = {lim:.2f} K (saddle-node point: the first maximum of Tc along "
              f"the steady-state curve, where f = 0 and df/dT = 0 for the energy balance f), {lim - top['Tc']:.2f} K "
              f"above the upper edge of the identified range. At {cfg['hot_tc']:.1f} K the steady states are: {hot_txt}"
              + (" — none of them stable, so the reactor ignites and cannot settle (section 7)."
                 if all(h["max_real_eig"] > 0 for h in hot) else "."),
              "", "The saddle-node point under changes of the process parameters:", "",
              f"| Parameters | Tc at the limit, K | Ca at the limit, mol/L | T at the limit, K | "
              f"Ca >= {ca_min:.2f} excludes the limit |", "|---|---|---|---|---|"]
        L += [f"| {r['case']} | {r['tc_limit']:.2f} | {r['ca_at_limit']:.4f} | {r['T_at_limit']:.1f} | "
              f"{'yes' if r['ca_min_excludes_limit'] else 'no'} |" for r in pr["limit_sensitivity"]]
        sens = pr["limit_sensitivity"]
        inside = [r["case"] for r in sens if r["tc_limit"] < top["Tc"]]
        fail = [r["case"] for r in sens if not r["ca_min_excludes_limit"]]
        Ts = [r["T_at_limit"] for r in sens]
        cr = pr["caf_rise_breaking_ca_min"]
        L += ["", "Neither candidate constraint follows the limit reliably:",
              f"- a fixed bound on Tc: the limit moves inside the identified range in "
              f"{len(inside)} of {len(sens)} cases ({', '.join(inside)});",
              f"- a lower bound Ca >= {ca_min:.2f} mol/L (the steady-state Ca at the edge of the data rounded up): "
              f"Ca at the limit grows with the feed concentration Caf. A rise of Caf by {100 * cr['nominal']:.2f} % "
              f"brings Ca at the limit to {ca_min:.2f} (by {100 * cr['UA -10 %']:.2f} % with 10 % less heat transfer); "
              "beyond that the bound admits inputs at which the cold steady state no longer exists"
              + (f" (cases: {', '.join(fail)})." if fail else "."),
              f"- T at the limit stays between {min(Ts):.1f} and {max(Ts):.1f} K in all cases; if the reactor "
              "temperature is measured, it is a candidate for a constraint (not tested here).",
              "", f"Ca is also a late indicator. Open-loop test: from the cold steady state 0.5 K below the limit the "
              f"input steps to {pr['ignition_delay'][0]['tc_step']:.1f} K (inside the allowed range); a given time after "
              f"the true Ca falls to {ca_min:.2f}, the input drops at once to {pr['cool_tc']:.1f} K, the strongest cooling "
              "in the range. Peak reactor temperature, K:", "",
              "| Case | Start Tc, K (Ca) | Ca reaches the bound after, min | "
              + " | ".join(f"delay {r['delay_min']:g} min" for r in pr["ignition_delay"][0]["runs"]) + " |",
              "|---|---|---|" + "---|" * len(pr["ignition_delay"][0]["runs"])]
        for c in pr["ignition_delay"]:
            L.append(f"| {c['case']} | {c['tc_start']:.2f} ({c['ca_start']:.3f}) | {_f(c['runs'][0]['t_cross_min'], 2)} | "
                     + " | ".join(f"{r['T_max']:.1f}" for r in c["runs"]) + " |")
        L += ["", "A delay of a few samples between Ca crossing the bound and full cooling decides whether the reactor "
              "ignites. The constraints for the controller are therefore an open design question of the next part. "
              "Requirements that follow from this report:",
              "- an input bound with a margin to the limit for the worst admissible set of process parameters, "
              "which has to be stated (the table above);",
              "- protective constraints fed with the raw measurement: the filter can replace up to "
              f"{nj - 1} samples of a fast fall by predictions before it recognises a process change (section 7);",
              "- the supervisor's penalty increase must not slow down protective moves: on the hot side it rises "
              "while Ca is already falling (section 7);",
              "- a closed-loop test on the cases of the table.", ""]

    # ---------------------------------------------------------------- 5. drift
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
              f"- a frozen filter on a drifting process rejects valid data (real time): static filter {_pr(a1['filter'])}; "
              f"filter refitted with the window {_pr(w1['filter'])}.",
              "", "![drift](fig3_drift.png)", ""]

    # ---------------------------------------------------------------- 6. cold excursion
    if "excursion" in s:
        ex = s["excursion"]
        a, b = ex["runs"]["Static model"], ex["runs"]["Sliding window"]
        lo, hi = ex["outside_range"]
        H = ex["horizon"]
        pt = lambda r: f"{100 * sum(r[1:]):.0f} %"
        L += ["## 6. Level 3: leaving the training range (cold side)", "",
              f"The coolant temperature moves to {lo:.0f}-{hi:.0f} K, never seen during start-up, and returns. "
              f"The supervisor receives the largest GP variance along the free-run of the model over {H} "
              f"samples ({H * ts:g} min) under the planned input; in this open-loop test the plan is "
              "the input schedule itself, in a predictive controller it is the candidate input sequence. "
              f"The trajectory starts at the current regressor, so an input change is first seen {ex['max_lead']} "
              "samples before it: that is the largest possible warning. "
              f"The supervisor is cautious above {ex['th1']:.4f} and conservative above {ex['th2']:.4f}; the penalty on "
              "control moves rises continuously up to twofold. The thresholds are a heuristic: th1 equals the selected "
              "regularisation lam (the noise variance of the equivalent GP, normalised units) and th2 = 3 th1, so they "
              f"move with lam; here lam is {st['lam'] / st['noise_var_normalised']:.1f} times the estimated sensor-noise "
              "variance in the same units, so the thresholds are not tied to the measured noise. The GP prior variance "
              "is fixed at 1, so the variance measures data coverage, not a calibrated error.", "",
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
              f"| Filter outside, real time: precision / recall | {a['filter_outside']['precision']:.2f} / "
              f"{a['filter_outside']['recall']:.2f} | {b['filter_outside']['precision']:.2f} / {b['filter_outside']['recall']:.2f} |",
              f"| Samples outside in filter jump mode | {a['jump_samples_outside']} | {b['jump_samples_outside']} |",
              "", "A frozen model stays uncertain for as long as the process is outside the range. The sliding window "
              "learns the neighbourhood of the current operating point within a few samples, so the variance at the "
              "current point alone soon reads normal. Normal then means only that the planned trajectory runs through "
              "data the window has seen; the probed variances show the model's coverage of other inputs. "
              "Because the supervisor checks the planned trajectory, it can warn before the input leaves the range and "
              f"before it returns, by at most {ex['max_lead']} samples.",
              "With the window frozen (static model) the filter cannot agree with the new data, so it stays in jump "
              "mode, where single spikes are caught by the model-free trend test.",
              "", "![supervisor](fig4_supervisor.png)", ""]

    # ---------------------------------------------------------------- 7. hot side
    if "hot_excursion" in s:
        hx = s["hot_excursion"]
        R = hx["runs"]
        th1 = s["excursion"]["th1"] if "excursion" in s else None
        k0 = hx["first_opportunity"]
        ca_min = s["process"]["ca_min_constraint"] if "process" in s else None
        L += ["## 7. Hot side: beyond the stability limit", "",
              f"The coolant temperature is held at {hx['tc_hot']:.1f} K, {hx['tc_hot'] - cfg['tc_range'][1]:g} K above "
              f"the identified range and beyond the saddle-node point, for {hx['n_hot']} samples "
              f"({hx['n_hot'] * ts:g} min, from sample 300), then returns.",
              "- The reactor ignites: "
              + (f"the true Ca falls below {ca_min:.2f} at sample {hx['first_sample_below_ca_min']}; " if ca_min else "")
              + f"it leaves the start-up range (below {hx['ca_min_startup']:.3f}) at sample "
              f"{hx['first_sample_below_startup_ca']}, {hx['first_sample_below_startup_ca'] - 300} samples after the step. "
              f"Ca falls to {hx['Ca_min']:.3f} and the reactor temperature rises from {hx['T_before']:.1f} to "
              f"{hx['T_max']:.1f} K. After the return it settles back on the cold branch (Ca {hx['Ca_end']:.3f}, "
              f"T {hx['T_end']:.1f} K).",
              f"- The planned trajectory first contains {hx['tc_hot']:.0f} K at sample {k0}. From there to the step, the "
              f"largest variance along it is {R['Static model']['var_h_before_ignition']:.4f} (static) and "
              f"{R['Sliding window']['var_h_before_ignition']:.4f} (window), against the cautious threshold {_f(th1)}. "
              "The variance measures the distance from the data, not the closeness of the stability limit, and "
              f"{hx['tc_hot'] - cfg['tc_range'][1]:g} K beyond the data that distance is small.", "",
              "| | Static model | Sliding window |", "|---|---|---|",
              "| Normal just before the first opportunity | " + " | ".join(
                  "yes" if R[n]["normal_before"] else "no" for n in ("Static model", "Sliding window")) + " |",
              "| First warning (not normal), sample | " + " | ".join(
                  _n(R[n]["first_non_normal"]) for n in ("Static model", "Sliding window")) + " |",
              "| First conservative regime (penalty x2), sample | " + " | ".join(
                  _n(R[n]["first_conservative"]) for n in ("Static model", "Sliding window")) + " |",
              "| Measurements replaced in real time during the step (by the model prediction, in jump mode by the "
              "trend extrapolation) and restored later | " + " | ".join(
                  (", ".join(map(str, R[n]["hidden_in_real_time"])) or "none") for n in ("Static model", "Sliding window"))
              + " |",
              "| Largest overstatement of Ca by those replacements (value passed on - measurement), mol/L | " + " | ".join(
                  _f(R[n]["hidden_max_overstatement"]) for n in ("Static model", "Sliding window")) + " |",
              "", "A warning only raises the move penalty; it does not stop the input. While Ca is falling, the filter "
              f"may replace up to {nj - 1} measurements in a row by a prediction (of the model, or of the trend in jump mode) "
              "before the run is recognised as a process change, and the conservative regime doubles the move penalty. Both act against a protective move, so the "
              "supervisor is not a substitute for hard constraints (section 4); section 8 shows how often it warns in time.",
              "", "![hot side](fig5_hot_side.png)", ""]

    # ---------------------------------------------------------------- 8. seeds
    if seeds:
        P = seeds["per_seed"]
        n = seeds["n_seeds"]
        L += [f"## 8. Spread over {n} seeds", "",
              f"All simulated experiments repeated with seeds 0-{n - 1} (new input sequences, noise, "
              "spikes and start-up for each seed; every experiment of every seed has its own random stream). "
              "Median [25th; 75th percentile]:", "",
              "| Quantity | Median [IQR] |", "|---|---|"]
        for key, label, nd, pct in SEED_METRICS:
            L.append(f"| {label} | {_spread([r.get(key) for r in P], nd, pct)} |")
        lead_min = 10                     # samples = 1 min
        hot = {}
        for cfg_name, key in (("sliding window", "window"), ("static model", "static")):
            lags = [r[f"hot_warning_lag_{key}"] for r in P]
            hot[cfg_name] = (sum(v is not None and v <= -lead_min for v in lags),
                             sum(not r[f"hot_normal_before_{key}"] for r in P),
                             sum(r[f"hot_first_warning_{key}"] == r["hot_first_opportunity"]
                                 and r[f"hot_normal_before_{key}"] for r in P))
        first_ret = sum(r["exc_window_lead_return"] == r["exc_max_lead"] for r in P)
        oe_better = sum(r["oe_over_krr"] < 1 for r in P)
        lams, counts = np.unique([r["lam"] for r in P], return_counts=True)
        worst_ratio = max(r["tune_worst_candidate_free_run"] / r["tune_best_candidate_free_run"] for r in P)
        n_excl = sum(r["tune_n_candidates"] - r["tune_n_admissible"] > 0 for r in P)
        L += ["", "Hot side: the supervisor warned at least "
              f"{lead_min} samples ({lead_min * ts:g} min) before Ca left the start-up range in "
              + "; ".join(f"{v[0]} of {n} seeds with the {k} ({v[2]} of them at the first opportunity; "
                          f"{_seeds(v[1])} not counted because it was already not normal before)" for k, v in hot.items())
              + ". Even when it warns, it only raises the move penalty.",
              f"Cold excursion, sliding window: the warning before the input returns came at the first opportunity "
              f"({seeds['per_seed'][0]['exc_max_lead']} samples before) in {first_ret} of {n} seeds.",
              "Selected regularisation (it also sets the supervisor thresholds): " + ", ".join(
                  f"{l:.1e} in {_seeds(c)}" for l, c in zip(lams, counts)) + ".",
              f"Tuning: the free-run stage removed candidates in {n_excl} of {n} seeds; the worst LOO candidate had "
              f"{worst_ratio:.0f} times the best held-out free-run RMSE. The chosen setting stays within "
              f"{100 * cfg['fr_tol']:.0f} % of the best by construction (largest ratio "
              f"{max(r['tune_chosen_over_best'] for r in P):.2f}).",
              f"The linear OE model was more accurate than the kernel model in {oe_better} of {n} seeds.", ""]

    # ---------------------------------------------------------------- 9-11
    c = st["checks"]
    sn = s["process"]["saddle_node_check"] if "process" in s else None
    L += [f"## {9 if seeds else 8}. Numerical self-checks", "",
          f"- Analytical gradient against central differences: largest relative deviation {c['gradient_max_rel_dev']:.1e} "
          f"(normalised units), {c['gradient_physical_max_rel_dev']:.1e} (physical units, as needed for MPC).",
          f"- Closed-form leave-one-out residuals against actual refits: largest deviation {c['loo_max_abs_dev']:.1e}; "
          f"the eigen-decomposition shortcut used for tuning against them: relative deviation {c['loo_grid_rel_dev']:.1e}.",
          f"- GP variance from the KRR factorisation against a library GP: largest deviation {c['variance_max_abs_dev']:.1e}.",
          f"- Leave-one-out search over {c['loo_grid_size']} settings on {c['loo_grid_n']} samples: {c['loo_grid_seconds']:.1f} s."]
    if sn:
        L.append(f"- Stability limit: at the computed saddle-node point |f| = {sn['f']:.1e} and |df/dT| = "
                 f"{sn['df_dT']:.1e} K/min (central differences).")
    L += ["- These checks confirm that the code matches the formulas; they do not show that the GP variance is a "
          "calibrated error estimate.",
          "", f"## {10 if seeds else 9}. Design choices for a dynamic process", "",
          f"- The model is dynamic (NARX with {cfg['lags'][0]} past outputs and {cfg['lags'][1]} past inputs), not a static input-output map.",
          f"- Anomaly threshold: calibrated to reject {100 * cfg['alpha']:.1f} % of normal samples on start-up data "
          "(out-of-sample residual quantile).",
          f"- Window of {cfg['window']} samples, refitted every {cfg['refit_every']} accepted samples; the SVR filter is refitted together with it.",
          f"- A run of {nj} rejections is taken as a process change: the rejected samples are restored to the history "
          "and the window; until the filter agrees again, single spikes are caught by a model-free trend test and "
          f"the models are refitted every {cfg['refit_every_jump']} accepted samples.",
          f"- The supervisor checks the trajectory planned over {cfg['horizon']} samples, not only the current point.",
          f"- Tuning: leave-one-out candidates within {100 * cfg['loo_tol']:.0f} % of the best, then admissible within "
          f"{100 * cfg['fr_tol']:.0f} % of the best free run on the held-out {100 * cfg['holdout']:.0f} % of the start-up "
          "block; among those, the strongest regularisation and then the narrowest kernel.",
          "- A rejected or missing sample is replaced by the model prediction in the regressor history.",
          "", f"## {11 if seeds else 10}. Limits", "",
          f"- Validated for Tc between {d['input_min']:.1f} and {d['input_max']:.1f} K at a sampling period of {ts:g} min.",
          "- Simulation study on a textbook process; no plant data.",
          "- The supervisor's penalty multiplier is computed but not yet used: there is no controller in this part.",
          "- Anomalies are isolated spikes only; a sensor fault that lasts several samples is treated as a process change.",
          ""]
    path.write_text("\n".join(L))
