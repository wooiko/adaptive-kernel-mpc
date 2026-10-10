"""Mutation check of part 2 (TZ section 10, rule P13): every mutation is a deliberate error in one
key function (a sign, an index shift, a lost term); the tests must catch each of them.

    python tests/mutate_mpc.py          # writes outputs/mutations_mpc.txt, exit code 1 if one survives

Each mutation is applied to a copy of the repository in a temporary directory, then
tests/test_mpc.py runs there; the mutation is caught if the run fails. A mutation whose source line
no longer matches exactly once is reported as stale (the list must follow the code).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mutations_mpc.txt"
TIMEOUT_S = 180          # the unmutated tests take under 10 s

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
]


def copy_repo(dst):
    for f in ROOT.glob("*.py"):
        shutil.copy2(f, dst / f.name)
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
            if src.count(old) != 1:
                verdict = "STALE"
                survived += 1
            else:
                (d / fname).write_text(src.replace(old, new), encoding="utf-8")
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
