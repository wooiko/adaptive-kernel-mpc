"""Diagnostic by the truth (P3: not part of any choice): why the adaptive K-MPC settles more slowly than the
static one in S3 (k0 drift), reference seed.

    python diagnose_mpc.py        # writes outputs/mpc_diag_s3.json (frozen settings of outputs/mpc_tuning.json)

Every EVERY samples of S3 the controller model is compared with the true plant at the current parameters:
  * steady-state gain dCa/dTc at the Tc that holds the current set-point (central differences, +-0.1 K),
    of the true cold steady state and of the model's own steady state y = f(y, y, u, u);
  * bias of the model steady state at that Tc;
  * free-run RMSE over a 15-min test input (+-1.5 K around that Tc) from the true steady state;
  * input excitation of the identifier's window: standard deviation and range of Tc in it.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import cstr  # noqa: E402
import data  # noqa: E402
import pipeline as pl  # noqa: E402
import run_mpc as rm  # noqa: E402
import safety  # noqa: E402

EVERY = 150             # samples between snapshots of the model
TEST_SEED = 5           # stream of the test input (diagnostic only)
NAMES = ("K-MPC static", "K-MPC adaptive")


class Snapshots:
    """Wraps a controller; keeps a copy of the identifier's window model every EVERY samples."""
    def __init__(self, ctrl, ident): self.c, self.ident, self.snaps = ctrl, ident, {}
    def planned_inputs(self, u_prev, n): return self.c.planned_inputs(u_prev, n)

    def step(self, k, y, u, r, info):
        if k % EVERY == 0:
            self.snaps[k] = (copy.deepcopy(self.ident.krr), np.array(self.ident.Z))
        return self.c.step(k, y, u, r, info)


def model_steady_state(f, u, y0, iters=400):
    y = y0
    for _ in range(iters):
        y = f(np.array([y, y, u, u]))
    return y


def diagnose(c, s, lim, name):
    sc = cl.scenario_s3(rm.REFERENCE_SEED, lim)
    ctrl, ident = rm.make(name, c, lim, s)
    w = Snapshots(ctrl, ident)
    log = cl.run_loop(sc["plant"], sc["x0"], ident, w, sc["r"], sc["add"], sc["missing"], sc["u0"],
                      supervisor=rm.CONFIGS[name][2])
    test_u = data.aprbs(150, -1.5, 1.5, 5, 20, np.random.default_rng(TEST_SEED))
    static = pl.StaticKRR(c)
    rows = []
    for k, (krr, Z) in sorted(w.snaps.items()):
        p, r = sc["plant"].params(k), sc["r"][k]
        tc = brentq(lambda t: safety.cold_steady_state(t, p)[0] - r, safety.TC_MIN - 3.0, lim + 0.4)
        g_true = (safety.cold_steady_state(tc + 0.1, p)[0] - safety.cold_steady_state(tc - 0.1, p)[0]) / 0.2
        if name == "K-MPC static":
            f = lambda x: float(static.predict(x)[0])
        else:
            fr = pl.Frozen(krr, c["scaler"])
            f = lambda x, fr=fr: float(fr.predict(x[None, :])[0])
        ys = model_steady_state(f, tc, r)
        g_mod = (model_steady_state(f, tc + 0.1, ys) - model_steady_state(f, tc - 0.1, ys)) / 0.2
        uu = np.clip(tc + test_u, safety.TC_MIN, lim)
        truth = cstr.simulate(uu, data.TS, np.array(safety.cold_steady_state(uu[0], p)), p)[:, 0]

        class M:
            predict = staticmethod(lambda X: np.array([f(x) for x in np.atleast_2d(X)]))
        sim = pl.free_run(M, truth, uu)
        u_win = Z[:, 2] * c["scaler"].sx[2] + c["scaler"].mx[2]
        rows.append({"k": int(k), "k0_mult": p.k0 / cstr.CSTRParams().k0, "setpoint": float(r), "tc_true": tc,
                     "gain_true": g_true, "gain_model": g_mod, "gain_ratio": g_mod / g_true,
                     "ss_bias": ys - r, "free_run_rmse": float(np.sqrt(np.mean((sim[2:] - truth[2:]) ** 2))),
                     "window_tc_std": float(np.std(u_win)), "window_tc_range": float(np.ptp(u_win))})
    seg = cl.segment_metrics(log["Ca"], log["r"], sc["starts"], len(log["r"]))
    return {"snapshots": rows, "settling_min": [x["settling_min"] for x in seg],
            "overshoot": [x["overshoot"] for x in seg], "segment_starts": sc["starts"]}


def main():
    lim = safety.tc_max()["tc_max"]
    s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
    with threadpool_limits(1):
        c, _ = rm.commissioning(rm.REFERENCE_SEED)
        out = {"seed": rm.REFERENCE_SEED, "scenario": "S3", "kind": "diagnostic by the truth (P3)",
               "settings": s, "every": EVERY, "results": {n: diagnose(c, s, lim, n) for n in NAMES}}
    (rm.OUT / "mpc_diag_s3.json").write_text(json.dumps(out, indent=2))
    for n in NAMES:
        q = out["results"][n]["snapshots"]
        print(n, "gain ratio", round(min(x["gain_ratio"] for x in q), 3), "..", round(max(x["gain_ratio"] for x in q), 3),
              "| bias max", round(max(abs(x["ss_bias"]) for x in q), 4),
              "| window Tc std", round(q[0]["window_tc_std"], 2), "->", round(q[-1]["window_tc_std"], 2))


if __name__ == "__main__":
    main()
