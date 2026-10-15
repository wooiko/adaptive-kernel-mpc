"""Mutation check of part 2 (TZ section 10, rule P13): every mutation is a deliberate error in one
key function (a sign, an index shift, a lost term); the tests must catch each of them.

    python tests/mutate_mpc.py          # writes outputs/mutations_mpc.txt, exit code 1 if one survives

Each mutation is applied to a copy of the repository in a temporary directory, then
tests/test_mpc.py runs there; the mutation is caught if the run fails. A mutation whose source line
no longer matches exactly once is reported as stale (the list must follow the code).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mutations_mpc.txt"
TIMEOUT_S = 180          # the unmutated tests take about 16 s (2 cores, one BLAS thread)

# (id, file, original, mutated, what it imitates)
MUTATIONS = [
    ("grad_scale", "adaptive.py",
     "return self.krr.gradient(self.sc.x(np.asarray(x, float))) * self.sc.st / self.sc.sx",
     "return self.krr.gradient(self.sc.x(np.asarray(x, float)))",
     "window-model gradient without the factor s_t/s_x"),
    ("G_column_shift", "mpc.py", "        G[i, i] += b1", "        G[i, min(i + 1, Np - 1)] += b1",
     "sensitivity: b1 in the column one step later"),
    ("G_sign_b2", "mpc.py", "            G[i, i - 1] += b2", "            G[i, i - 1] -= b2",
     "sensitivity: sign of b2"),
    ("regressor_u_lag", "mpc.py", "        X[i] = (y0, y1, Ubar[i], up)", "        X[i] = (y0, y1, up, up)",
     "nominal trajectory: regressor with u_k-1 instead of u_k"),
    ("d_off", "mpc.py", "        y_next = model.f(X[i]) + d", "        y_next = model.f(X[i])",
     "disturbance estimate d_k switched off"),
    ("blocking_hold", "mpc.py", "np.minimum(np.arange(Np), Nc - 1)", "np.minimum(np.arange(Np), Nc - 2)",
     "move blocking: input held one move early"),
    ("no_shift", "mpc.py", "    return np.r_[V[1:], V[-1]]", "    return V",
     "nominal plan not shifted"),
    ("q_no_u_prev", "mpc.py", "        q = gy - 2.0 * D.T @ W @ e * u_km1", "        q = gy",
     "QP: term e u_k-1 lost in q"),
    ("rate_first_row", "mpc.py", "    rate_l, rate_u = -du_max + e * u_km1, du_max + e * u_km1",
     "    rate_l, rate_u = -du_max + 0 * e, du_max + 0 * e",
     "QP: first rate row without u_k-1"),
    ("mult_on_cooling", "mpc.py", "[Z, Z, 2.0 * r0 * I]", "[Z, Z, 2.0 * mult * r0 * I]",
     "supervisor multiplier also on cooling moves"),
    ("q_sign", "mpc.py", "qp.q + qp.P @ s", "-qp.q + qp.P @ s",
     "solver: sign of q"),
    ("accept_unsolved", "mpc.py", 'if status != "solved" or res.x is None', "if res.x is None",
     "solver: result used without status 'solved'"),
    ("fallback_unclipped", "mpc.py",
     '"V": clip_plan(Vbar, u_km1, u_min, u_max, du_max), "status": status, "fallback": True',
     '"V": Vbar, "status": status, "fallback": True',
     "fallback plan not clipped to the constraints"),
    ("tcmax_max", "safety.py", "    worst, v = min(lims, key=lambda t: t[0])",
     "    worst, v = max(lims, key=lambda t: t[0])", "Tc_max: maximum instead of minimum over the vertices"),
    ("tcmax_margin_sign", "safety.py", "    out = worst - margin", "    out = worst + margin",
     "Tc_max: margin with the sign +"),
    ("loop_input_delay", "closed_loop.py", "X[k] = plant.step(X[k - 1], u[k - 1], k - 1)",
     "X[k] = plant.step(X[k - 1], u[max(k - 2, 0)], k - 1)", "closed loop: plant driven by u_k-1 instead of u_k"),
    ("d_regressor_lag", "mpc.py", "x_km1 = np.array([y[k - 1], y[k - 2], u[k - 1], u[k - 2]], float)",
     "x_km1 = np.array([y[k - 1], y[k - 2], u[k - 2], u[k - 2]], float)",
     "K-MPC: regressor of d_k with u_k-2 instead of u_k-1"),
    ("ctrl_raw_measurement", "closed_loop.py", "u[k], det = controller.step(k, y_hist, u, r[k], info)",
     "u[k], det = controller.step(k, np.where(np.isnan(y_meas), y_hist, y_meas), u, r[k], info)",
     "closed loop: controller fed the raw measurement instead of y_rt"),
    ("restore_dropped", "closed_loop.py", "            y_hist[idx] = y_meas[idx]\n",
     "            pass\n", "closed loop: restored run not put back into the history"),
    ("d_on_missing", "mpc.py", 'hold = info.get("missing", False) or', "hold = False or",
     "K-MPC: d_k recomputed on a missing sample"),
    ("settle_window_early", "closed_loop.py", "out = np.abs(ca[s:e] - r[s:e]) > band",
     "out = np.abs(ca[s - 5:e] - r[s - 5:e]) > band", "metrics: settling search starts before the event"),
    ("d_filter_bypass", "mpc.py", "self.d = self.d + self.beta * (e - self.d)", "self.d = e",
     "K-MPC: filter of the disturbance estimate bypassed"),
    ("hold_on_reject_off", "mpc.py", "hold_d_on_reject=True, solver_settings=None",
     "hold_d_on_reject=False, solver_settings=None", "K-MPC: d_k recomputed on a rejected sample by default"),
    ("supervisor_ignored", "closed_loop.py", '"mult": res["mult"] if supervisor else 1.0', '"mult": res["mult"]',
     "closed loop: supervisor switch ignored"),
    ("pid_no_backcalc", "controllers.py",
     "self.I += self.Kc * self.ts / self.tau_i * e + self.ts / self.tau_i * (uk - v)",
     "self.I += self.Kc * self.ts / self.tau_i * e", "PID: anti-windup back-calculation removed"),
    ("nmpc_grad_sign", "controllers.py", "+ mpc.difference_matrix(Nc).T @ (2.0 * w * dU)",
     "- mpc.difference_matrix(Nc).T @ (2.0 * w * dU)", "NMPC: sign of the move term in the gradient"),
    ("anchor_refit_dropped", "adaptive.py", "            self.krr.fit(*self._rows())\n        else:",
     "            self.krr.fit(np.array(self.Z), np.array(self.tz))\n        else:",
     "anchored window: fixed rows dropped at a refit"),
    ("anchor_newest", "adaptive.py", "self._Za, self._tza = Zw[:anchor], tw[:anchor]",
     "self._Za, self._tza = Zw[len(Zw) - anchor:], tw[len(tw) - anchor:]",
     "anchored window: fixed rows taken from the newest instead of the oldest"),
    ("gate_sign", "mpc.py", "    return k is not None and k < 0.0", "    return k is not None and k > 0.0",
     "sign gate: sign of the gain test"),
    ("gate_ignored", "adaptive.py", "            if self.refit_check(self, cand):", "            if True:",
     "sign gate: a rejected refit applied anyway"),
    ("learn_window_short", "closed_loop.py", "k - last_change <= learn_after_change",
     "k - last_change < learn_after_change", "excitation window: one sample short"),
    ("learning_ignored", "adaptive.py", "            if self.adapt and self.learning:\n                self.Z.append(z[0])",
     "            if self.adapt:\n                self.Z.append(z[0])", "excitation window: learning flag ignored"),
    ("r2_last_rows", "run_mpc.py", "np.linspace(0, len(c[\"Z\"]) - 1, window)",
     "np.arange(len(c[\"Z\"]) - window, len(c[\"Z\"]))", "initial window from the last rows of the record"),
    ("readme_number_edit", "README.md", r"re:(\| S1, set-point steps: IAE, mol/L min \| \d+\.)(\d)",
     r"\g<1>9\g<2>", "README: a number of the Part 2 table edited by hand"),
    ("plan_unshifted", "mpc.py", "return (blocking_matrix(n + 1, self.Nc) @ self.V)[1:]",
     "return (blocking_matrix(n + 1, self.Nc) @ self.V)[:-1]", "supervisor: controller's plan not shifted by one step"),
    ("viol_rate_dropped", "closed_loop.py", " + np.sum(np.abs(du) > du_max + tol))", ")",
     "metrics: moves above du_max not counted as violations"),
    ("ignition_last_T", "closed_loop.py", '"ignition": bool(np.nanmax(T) > IGNITION_T)',
     '"ignition": bool(T[-1] > IGNITION_T)', "metrics: ignition judged on the last T instead of the highest"),
    ("fallback_scope_s5b", "report_mpc.py", "        for sc in IN_SET:\n            for _, m in runs(S, sc, conf):",
     "        for sc in SCENARIOS:\n            for _, m in runs(S, sc, conf):",
     "acceptance: fallback steps of S5b counted in the D11 share"),
    ("viol_scope_in_set", "report_mpc.py", 'viol = sum(m["violations"] for sc in SCENARIOS',
     'viol = sum(m["violations"] for sc in IN_SET', "acceptance: violations counted inside the admissible set only"),
    ("grid_edge_index", "run_mpc.py", 'return "lower edge" if on[0] == 0 else', 'return "lower edge" if on[0] == 1 else',
     "tuning: lower edge of the grid not recognised"),
    ("full_ms_ctrl_only", "closed_loop.py",
     '"full_ms": {"median": float(np.median(tc + ti)), "p95": float(np.percentile(tc + ti, 95)),',
     '"full_ms": {"median": float(np.median(tc)), "p95": float(np.percentile(tc, 95)),',
     "step time: the whole sample without the identifier (D15)"),
    ("step_median_ignored", "report_mpc.py", 'all(d["A_median_ms"] <= STEP_LIMIT_MS', 'all(min(d["A_by_pass_ms"]) <= STEP_LIMIT_MS',
     "acceptance: step time decided on the best pass instead of the median (D15)"),
    ("step_non_kmpc", "report_mpc.py", 'if c.startswith("K-MPC")}', 'if True}',
     "acceptance: step time of the NMPC counted as K-MPC"),
    ("pass_drift_ignored", "measure_step_time.py", "lost_share <= LOST_MAX and drift <= BENCH_DRIFT_MAX",
     "lost_share <= LOST_MAX", "step time: benchmark drift not checked (D15)"),
    ("decide_invalid_used", "measure_step_time.py", '    valid = [p for p in passes if p["valid"]]', "    valid = passes",
     "step time: invalid passes in the median (D15)"),
    ("pooled_wall", "measure_step_time.py", "cpu, wall, ref = R[:, 0] + R[:, 2], R[:, 1] + R[:, 3]",
     "cpu, wall, ref = R[:, 1] + R[:, 3], R[:, 0] + R[:, 2]", "step time: wall time pooled as CPU time"),
    ("clock_wall_as_cpu", "measure_step_time.py", "c0, w0 = time.thread_time(), time.perf_counter()",
     "c0, w0 = time.perf_counter(), time.perf_counter()", "step time: CPU clock replaced by the wall clock"),
    ("tie_exact", "report_mpc.py", "    eq = ties(A, B)", "    eq = A == B", "pairs: ties by exact equality (audit 1, C3)"),
    ("fingerprint_no_settings", "run_mpc.py", '    h.update(json.dumps(json.loads((root / "outputs" / "mpc_tuning.json")',
     '    len(json.dumps(json.loads((root / "outputs" / "mpc_tuning.json")',
     "provenance: frozen settings left out of the code fingerprint (audit 1, C1)"),
    ("diag_den_ignored", "diagnose_offset.py", "    k = mpc.ss_gain(g)\n    return np.nan if k is None else k",
     "    a1, a2, b1, b2 = g\n    return float((b1 + b2) / (1.0 - a1 - a2))",
     "S0 diagnostic: gain without the check of 1 - a1 - a2 (audit 1, N3)"),
    ("tune_ss_ignored", "run_mpc.py", 'r.get("violations", 0) == 0 and r["ss_max"] <= SS_LIMIT]',
     'r.get("violations", 0) == 0]', "tuning: steady-state limit ignored in the selection"),
    ("ref_ignition_free", "check_reference_run.py", 'EXACT_RUN = ("violations", "ignition")', 'EXACT_RUN = ("violations",)',
     "reference run: ignition not compared exactly (D16)"),
    ("ref_factor_ignored", "check_reference_run.py", "            tol = factor * env[sc][k]", '            tol = float("inf")',
     "reference run: tolerance factor ignored (D16)"),
    ("ref_none_unpaired", "check_reference_run.py", '                return float("inf")\n            continue',
     "                continue", "reference run: None paired with a number (D16)"),
    ("ref_s5b_bounded", "check_reference_run.py", 'IN_SET = ("S1", "S2", "S3", "S4", "S5a", "S6")',
     'IN_SET = ("S1", "S2", "S3", "S4", "S5a", "S5b", "S6")', "reference run: S5b under the tolerance (D16)"),
    ("ref_time_printed", "check_reference_run.py", '"wall_s", "wall_s_total", "machine"}', '"wall_s_total"}',
     "reference run: time fields compared (D16)"),
]


def copy_repo(dst):
    for f in ROOT.glob("*.py"):
        shutil.copy2(f, dst / f.name)
    shutil.copy2(ROOT / "README.md", dst / "README.md")
    (dst / "outputs").mkdir()
    for f in ROOT.glob("outputs/*"):          # result files read by the tests (P4, P12)
        if f.suffix in (".json", ".md"):
            shutil.copy2(f, dst / "outputs" / f.name)
    (dst / "tests").mkdir()
    for f in (ROOT / "tests").glob("*.py"):
        shutil.copy2(f, dst / "tests" / f.name)


def run_tests(cwd):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "unittest", "tests.test_mpc"], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False                     # a run that no longer finishes counts as failed: the mutation is caught
    return r.returncode == 0


def main():
    lines, survived = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"; base.mkdir(); copy_repo(base)
        if not run_tests(base):
            sys.exit("tests fail without mutations: fix them first")
        for mid, fname, old, new, what in MUTATIONS:
            d = Path(tmp) / mid; d.mkdir(); copy_repo(d)
            src = (d / fname).read_text(encoding="utf-8")
            if old.startswith("re:"):             # regular expression: the first match is changed
                n_match = len(re.findall(old[3:], src))
                src_new = re.sub(old[3:], new, src, count=1)
            else:
                n_match, src_new = src.count(old), src.replace(old, new)
            if n_match < 1 or (not old.startswith("re:") and n_match != 1):
                verdict = "STALE"
                survived += 1
            else:
                (d / fname).write_text(src_new, encoding="utf-8")
                verdict = "survived" if run_tests(d) else "caught"
                survived += verdict == "survived"
            lines.append(f"{verdict:8s}  {mid:20s}  {fname:12s}  {what}")
            print(lines[-1], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("Mutations of part 2 (tests/mutate_mpc.py); the tests of tests/test_mpc.py must catch each.\n"
                   + "\n".join(lines) + f"\n{len(MUTATIONS) - survived} of {len(MUTATIONS)} caught\n",
                   encoding="utf-8")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
