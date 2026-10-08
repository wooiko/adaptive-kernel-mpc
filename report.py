"""Fills the validation report from the computed numbers (no hand-typed results)."""


def _pr(p):
    return f"{p['precision']:.2f} / {p['recall']:.2f} ({p['caught']} of {p['true_anomalies']} caught, {p['flagged']} flagged)"


def write_report(s, path):
    d, st, cfg = s["data"], s["stationary"], s["settings"]
    arx, krr = st["models"]["ARX (linear)"], st["models"]["Kernel ridge"]
    L = ["# Adaptive kernel model of a CSTR: validation report", "",
         "Process: continuous stirred-tank reactor, exothermic reaction A -> B. "
         "Input: coolant temperature Tc. Output: concentration Ca. All errors are in mol/L.",
         ("Data: simulated experiments on the textbook model (Henson & Seborg, 1997) with sensor noise, "
          "isolated spikes in 2 % of the samples and a logger gap added on purpose."
          if s["simulated"] else "Data: supplied log file."), "",
         "The identification system has three levels that share one RBF kernel:", "",
         "1. an SVR filter rejects anomalous measurements;",
         f"2. kernel ridge regression (KRR) on a sliding window of {cfg['window']} samples is the process model;",
         "3. a Gaussian-process (GP) supervisor measures how far the operating point is from the training data.",
         "", "## 1. Data", "",
         f"- Samples: {d['n_samples']}, sampling period {s['sampling_min']} min; missing values: {d['n_missing']}.",
         f"- Estimated sensor noise: {st['noise_std_est']:.4f} (standard deviation).",
         f"- Input range: {d['input_min']:.1f} to {d['input_max']:.1f} K, {d['input_levels']} distinct levels; "
         f"the least-visited tenth of the range holds {100 * d['input_emptiest_decile_share']:.1f} % of samples.",
         f"- Start-up (cleaning, tuning, first model): first {st['n_startup']} samples. "
         f"Test: last {st['n_test']} samples, never used for fitting or tuning.",
         "", "## 2. Level 1: anomaly filter", "",
         f"A measurement is rejected when it differs from the SVR one-step prediction by more than "
         f"{cfg['kappa']} standard deviations of the normal one-step scatter. "
         f"On the test data the filter rejected {st['filter']['n_flagged_test']} samples.", ""]
    f = st["filter"]
    if "sweep" in f:
        L += ["Precision / recall against the known injected spikes, test data:", "",
              "| Threshold | SVR filter | Rolling median (reference) |", "|---|---|---|"]
        L += [f"| {r['kappa']} | {_pr(r['svr'])} | {_pr(r['rolling_median'])} |" for r in f["sweep"]]
        L += ["", f"The rows other than {cfg['kappa']} were run with the window frozen.",
              f"With the filter the one-step error of the window model is {f['one_step_rmse_with_filter']:.4f}; "
              f"without it, {f['one_step_rmse_without_filter']:.4f} "
              f"({f['one_step_rmse_without_filter'] / f['one_step_rmse_with_filter']:.1f} times larger).",
              "For isolated spikes on a smooth signal a plain rolling median detects as well as the model-based "
              "filter or better; the model-based filter does not rely on the signal being smooth in time."]
    L += ["", "![filter](fig1_filter.png)", "",
          "## 3. Level 2: process model", "",
          "KRR and the linear reference (ARX) predict the next sample of Ca from the last two samples of Ca and Tc. "
          f"Kernel width {st['sigma']:.2f} and regularisation {st['lam']:.4f} (normalised units) were chosen by exact "
          "leave-one-out cross-validation.",
          f"Errors are measured against the {st['reference']}. "
          "Free-run: the model is fed its own predictions, as inside a controller.", "",
          "| Model | One-step RMSE | Free-run RMSE | Free-run max error | Free-run fit |", "|---|---|---|---|---|"]
    for name, m in st["models"].items():
        L.append(f"| {name} | {m['one_step_rmse']:.4f} | {m['free_run_rmse']:.4f} | "
                 f"{m['free_run_max_abs_err']:.4f} | {m['free_run_fit_percent']:.1f} % |")
    L += ["", f"The kernel model's free-run error is {arx['free_run_rmse'] / krr['free_run_rmse']:.1f} times smaller "
          "than the linear model's.",
          f"The error is not uniform: where Ca is in its lowest tenth (below {st['ca_low_threshold']:.3f}, the hot end, "
          f"where the reactor gain is steepest) the kernel model's free-run RMSE is {st['krr_free_run_rmse_low_ca']:.4f}; "
          f"elsewhere it is {st['krr_free_run_rmse_elsewhere']:.4f}.",
          "", "![free run](fig2_free_run.png)", ""]
    if "drift" in s:
        dr = s["drift"]
        static = dr["strategies"]["Static model, no filter"]["rmse"]
        full = dr["strategies"]["SVR filter + sliding window"]["rmse"]
        L += ["## 4. Slow drift of the process", "",
              f"Catalyst activity falls linearly to {100 * dr['k0_end']:.0f} % over {dr['n_online']} samples. "
              "One-step RMSE against the true output, by quarter of the run:", "",
              "| Strategy | Q1 | Q2 | Q3 | Q4 | Whole run |", "|---|---|---|---|---|---|"]
        for name, r in dr["strategies"].items():
            L.append(f"| {name} | " + " | ".join(f"{q:.4f}" for q in r["rmse_by_quarter"]) + f" | {r['rmse']:.4f} |")
        L += ["", f"The full system is {static / full:.1f} times more accurate than the static model. "
              f"Filter precision / recall during the drift: {_pr(dr['filter'])}.",
              "", "![drift](fig3_drift.png)", ""]
    if "excursion" in s:
        ex = s["excursion"]
        a, b = ex["runs"]["Static model"], ex["runs"]["Sliding window"]
        lo, hi = ex["outside_range"]
        L += ["## 5. Level 3: leaving the training range", "",
              f"The coolant temperature moves to {lo:.0f}-{hi:.0f} K, never seen during start-up, and returns. "
              f"The supervisor is cautious when the GP variance exceeds {ex['th1']:.4f} and conservative above "
              f"{ex['th2']:.4f}; the penalty on control moves is then raised up to twofold.", "",
              "| | Static model | Sliding window |", "|---|---|---|",
              f"| GP variance inside the range (median) | {a['var_median_inside']:.5f} | {b['var_median_inside']:.5f} |",
              f"| GP variance outside (median) | {a['var_median_outside']:.5f} | {b['var_median_outside']:.5f} |",
              f"| Time outside spent cautious or conservative | {100 * sum(a['regime_share_outside'][1:]):.0f} % | "
              f"{100 * sum(b['regime_share_outside'][1:]):.0f} % |",
              f"| Time inside spent cautious or conservative | {100 * sum(a['regime_share_inside'][1:]):.1f} % | "
              f"{100 * sum(b['regime_share_inside'][1:]):.1f} % |",
              f"| One-step RMSE inside | {a['one_step_rmse_inside']:.4f} | {b['one_step_rmse_inside']:.4f} |",
              f"| One-step RMSE outside | {a['one_step_rmse_outside']:.4f} | {b['one_step_rmse_outside']:.4f} |",
              "", "A frozen model stays uncertain for as long as the process is outside the range. "
              "With the sliding window the model relearns the new region and the supervisor returns to normal.",
              "", "![supervisor](fig4_supervisor.png)", ""]
    c = st["checks"]
    L += ["## 6. Numerical self-checks", "",
          f"- Analytical gradient against central differences: largest relative deviation {c['gradient_max_rel_dev']:.1e}.",
          f"- Closed-form leave-one-out residuals against actual refits: largest deviation {c['loo_max_abs_dev']:.1e}.",
          f"- GP variance from the KRR factorisation against a library GP: largest deviation {c['variance_max_abs_dev']:.1e}.",
          f"- Leave-one-out search over {c['loo_grid_size']} settings on {c['loo_grid_n']} samples: {c['loo_grid_seconds']:.1f} s.",
          "", "## 7. Design choices for a dynamic process", "",
          f"- The model is dynamic (NARX with {cfg['lags'][0]} past outputs and {cfg['lags'][1]} past inputs), not a static input-output map.",
          f"- Anomaly threshold {cfg['kappa']} standard deviations: the one-step residual contains model error as well as sensor noise.",
          f"- Window of {cfg['window']} samples, refitted every {cfg['refit_every']} accepted samples; the SVR filter is refitted together with it.",
          "- The leave-one-out surface is flat near its minimum; among settings within 2 % of the best, the strongest "
          "regularisation and then the narrowest kernel are taken.",
          "- A rejected or missing sample is replaced by the model prediction in the regressor history.",
          "", "## 8. Limits", "",
          f"- Validated for Tc between {d['input_min']:.1f} and {d['input_max']:.1f} K at a sampling period of {s['sampling_min']} min.",
          "- Simulation study on a textbook process; no plant data.",
          "- The supervisor's penalty multiplier is computed but not yet used: there is no controller in this part.",
          "- Anomalies are isolated spikes only; a sensor fault that lasts several samples is treated as a process change.",
          ""]
    path.write_text("\n".join(L))
