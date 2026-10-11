"""Independent calculations of TZ section 10 for part 2 (another method than the code under test):

1. Loss of the linearisation: in states of the closed loop (adaptive K-MPC, S1, reference seed) the cost J of
   TZ 5.5 / 6.2 is evaluated on the nonlinear free run of the same window KRR for the K-MPC plan, for the plan of
   the code's NMPC (SLSQP with the analytic gradient through G) and for a reference optimum found by another
   solver (SciPy trust-constr with a numerical gradient, several starts). J is written out here again, without
   mpc.nominal or controllers.nonlinear_cost.
2. Ignition in S5a and S5b (reference seed): the inputs applied in each run are integrated again with LSODA from
   the same initial state and parameters; the highest T and the ignition verdict are compared with the run's.

    python check_mpc_independent.py      # writes outputs/mpc_independent.json (a few minutes, 2 processes)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, minimize  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import controllers as ctl  # noqa: E402
import cstr  # noqa: E402
import data  # noqa: E402
import mpc  # noqa: E402
import run_mpc as rm  # noqa: E402
import safety  # noqa: E402

SEED = rm.REFERENCE_SEED
EVERY = 20              # samples between the checked states of S1
STARTS = 4              # trust-constr starts: constant u_k-1, the K-MPC plan, two random feasible plans
LSODA_RTOL, LSODA_ATOL = 1e-8, 1e-10


def cost(model_f, y_k, y_km1, u_km1, d, V, r, q0, r0, mult, Np):
    """J of TZ 5.5 / 6.2 written out again: free run y_{j+1} = f(y_j, y_j-1, u_j, u_j-1) + d with the input held
    after the last move, set-point r over the horizon, move cost m r0 dU^2 for heating and r0 dU^2 for cooling."""
    U = [V[min(i, len(V) - 1)] for i in range(Np)]
    y0, y1, up, J = y_k, y_km1, u_km1, 0.0
    for i in range(Np):
        y_next = model_f(np.array([y0, y1, U[i], up])) + d
        J += q0 * (y_next - r) ** 2
        y0, y1, up = y_next, y0, U[i]
    dU = np.diff(np.r_[u_km1, V])
    return J + float(np.sum(np.where(dU > 0, mult * r0, r0) * dU ** 2))


class Probe:
    """Wraps the adaptive K-MPC; at every EVERY-th step compares its plan with the code's NMPC and a reference
    optimum of J. Everything else is delegated, so the loop runs as in run_mpc.py."""
    def __init__(self, inner):
        self.inner, self.rows, self.rng = inner, [], np.random.default_rng(7)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, k, y, u, r, info=None):
        info = info or {}
        v_prev = None if self.inner.V is None else self.inner.V.copy()
        out = self.inner.step(k, y, u, r, info)
        if k % EVERY == 0 and k > 2 * EVERY:
            self.rows.append(self.compare(k, y, u, r, info.get("mult", 1.0), v_prev))
        return out

    def compare(self, k, y, u, r, mult, v_prev):
        km, Nc, Np = self.inner, self.inner.Nc, self.inner.Np
        u1 = float(u[k - 1])
        args = (km.model.f, float(y[k]), float(y[k - 1]), u1, km.d)
        J = lambda V: cost(*args, V, r, km.q0, km.r0, mult, Np)   # noqa: E731
        nm = ctl.NMPC(km.model, km.u_max, km.rho, km.beta, Np=Np, Nc=Nc)
        nm.d = km.d
        v_n = nm.optimise(k, y, u, r, mult, mpc.shift_plan(v_prev, u1, Nc))["V"]
        D = mpc.difference_matrix(Nc)
        e = np.zeros(Nc); e[0] = u1
        lin = LinearConstraint(D, -km.du_max + e, km.du_max + e)
        box = Bounds(np.full(Nc, km.u_min), np.full(Nc, km.u_max))
        starts = [np.full(Nc, u1), km.V.copy()]
        while len(starts) < STARTS:
            starts.append(mpc.clip_plan(u1 + np.cumsum(self.rng.uniform(-km.du_max, km.du_max, Nc)), u1,
                                        km.u_min, km.u_max, km.du_max))
        best = None
        for v0 in starts:
            res = minimize(J, v0, method="trust-constr", constraints=[lin], bounds=box,
                           options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 3000})
            v = mpc.clip_plan(res.x, u1, km.u_min, km.u_max, km.du_max)
            if best is None or J(v) < J(best):
                best = v
        jr = J(best)
        return {"k": k, "J_kmpc": J(km.V), "J_nmpc": J(v_n), "J_ref": jr, "u0_kmpc": float(km.V[0]),
                "u0_ref": float(best[0])}


def linearisation_job(_):
    with threadpool_limits(1):
        lim = safety.tc_max()["tc_max"]
        c, _ = rm.commissioning(SEED)
        s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
        sc = cl.SCENARIOS["S1"](SEED, lim)
        ctrl, ident = rm.make("K-MPC adaptive", c, lim, s)
        probe = Probe(ctrl)
        cl.run_loop(sc["plant"], sc["x0"], ident, probe, sc["r"], sc["add"], sc["missing"], sc["u0"], supervisor=True)
        return probe.rows


def lsoda_job(args):
    scen, name = args
    with threadpool_limits(1):
        lim = safety.tc_max()["tc_max"]
        c, _ = rm.commissioning(SEED)
        s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
        _, _, m, lg = rm.job((c, SEED, lim, scen, name, s, mpc.NP, True))
        sc = cl.SCENARIOS[scen](SEED, lim)
        u, T_code = np.asarray(lg["u"]), np.asarray(lg["T"])
        x = np.asarray(sc["x0"], float)
        T = [x[1]]
        for k in range(1, len(u)):
            sol = solve_ivp(cstr.rhs, (0.0, data.TS), x, args=(u[k - 1], sc["plant"].params(k - 1)), method="LSODA",
                            rtol=LSODA_RTOL, atol=LSODA_ATOL)
            x = sol.y[:, -1]
            T.append(x[1])
        T = np.array(T)
        return {"scenario": scen, "config": name, "max_T_code": float(np.max(T_code)), "max_T_lsoda": float(np.max(T)),
                "max_abs_diff_T": float(np.max(np.abs(T - T_code))),
                "same_verdict": bool((np.max(T) > cl.IGNITION_T) == (np.max(T_code) > cl.IGNITION_T)),
                "ignition_code": bool(m["ignition"])}


def main():
    s5 = [(scen, name) for scen in ("S5a", "S5b") for name in rm.PLAN[scen]]
    with ProcessPoolExecutor(rm.WORKERS) as ex:
        fut_lin = ex.submit(linearisation_job, None)
        s5_rows = list(ex.map(lsoda_job, s5))
        rows = fut_lin.result()
    jr = np.array([r["J_ref"] for r in rows])
    loss = (np.array([r["J_kmpc"] for r in rows]) - jr) / jr
    gap = (np.array([r["J_nmpc"] for r in rows]) - jr) / jr
    du0 = np.abs(np.array([r["u0_kmpc"] - r["u0_ref"] for r in rows]))
    res = {"linearisation": {"scenario": "S1", "seed": SEED, "every": EVERY, "n_states": len(rows), "starts": STARTS,
                             "reference_method": "SciPy trust-constr",
                             "kmpc_loss_median": float(np.median(loss)), "kmpc_loss_max": float(np.max(loss)),
                             "nmpc_gap_median": float(np.median(gap)), "nmpc_gap_max": float(np.max(gap)),
                             "du0_median": float(np.median(du0)), "du0_max": float(np.max(du0)), "states": rows},
           "ignition_lsoda": {"seed": SEED, "rtol": LSODA_RTOL, "atol": LSODA_ATOL, "runs": s5_rows}}
    (rm.OUT / "mpc_independent.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res["linearisation"].items() if k != "states"}, indent=2))
    print(all(r["same_verdict"] for r in s5_rows))


if __name__ == "__main__":
    main()
