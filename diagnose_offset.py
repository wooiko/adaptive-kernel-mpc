"""Diagnostic by the truth (P3) of the steady-state offset of the adaptive K-MPC, on the tuning scenario S0 only.

S2 (a test scenario) showed a steady-state offset of the adaptive K-MPC after the UA ramp on 8 of 20 seeds. This
script looks for the same behaviour on S0, whose disturbance part has the same structure (a Tf step, a UA ramp to
-5 %, 15 min of constant disturbance after each), over seeds 0-19, so that a remedy can be designed and tuned without
test data. Every PROBE_EVERY samples it records:
  * the steady-state gain dCa/dTc of the controller's model at the current point, K_model = (b1 + b2) / (1 - a1 - a2)
    with [a1, a2, b1, b2] the model gradient at x = [y, y, u, u] (y: real-time measurement, u: last input);
  * the true steady-state gain at the same Tc with the plant parameters of that moment (central difference of the
    cold steady state, +-GAIN_STEP K);
  * the standard deviation of Tc over the rows of the identifier's window.

    python diagnose_offset.py             # writes outputs/mpc_diag_offset.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import run_mpc as rm  # noqa: E402
import safety  # noqa: E402

SEEDS = range(20)
CONFIGS = tuple(sys.argv[1].split(";")) if len(sys.argv) > 1 else (
    "K-MPC static", "K-MPC adaptive", "K-MPC adaptive, anchored window")
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "mpc_diag_offset.json"
PROBE_EVERY = 10
GAIN_STEP = 0.05     # K, central difference of the true steady-state gain


def ss_gain(g):
    a1, a2, b1, b2 = g
    return float((b1 + b2) / (1.0 - a1 - a2))


def true_gain(u, p):
    return float((safety.cold_steady_state(u + GAIN_STEP, p)[0] - safety.cold_steady_state(u - GAIN_STEP, p)[0])
                 / (2 * GAIN_STEP))


class Probe:
    def __init__(self, inner, ident, plant):
        self.inner, self.ident, self.plant, self.rows = inner, ident, plant, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, k, y, u, r, info=None):
        if k % PROBE_EVERY == 0 and k > 2:
            x = np.array([y[k], y[k], u[k - 1], u[k - 1]], float)
            km = ss_gain(self.inner.model.g(x))
            kt = true_gain(float(u[k - 1]), self.plant.params(k))
            Z = np.array(self.ident._rows()[0]) * self.ident.sc.sx + self.ident.sc.mx
            self.rows.append((k, km, kt, float(np.std(Z[:, 2]))))
        return self.inner.step(k, y, u, r, info)


def run_seed(seed):
    with threadpool_limits(1):
        lim = safety.tc_max()["tc_max"]
        c, _ = rm.commissioning(seed)
        return [run(seed, name, c, lim) for name in CONFIGS]


def run(seed, name, c, lim):
    with threadpool_limits(1):
        s = json.loads(rm.TUNING_FILE.read_text())["chosen"]
        sc = cl.scenario_s0(seed, lim)
        ctrl, ident = rm.make(name, c, lim, s)
        probe = Probe(ctrl, ident, sc["plant"])
        log = cl.run_loop(sc["plant"], sc["x0"], ident, probe, sc["r"], sc["add"], sc["missing"], sc["u0"],
                          supervisor=rm.CONFIGS[name][2])
        m = cl.run_metrics(log, sc["starts"], safety.TC_MIN, lim, 1.0)
        hold = sc["hold_segments"]
        rows = np.array(probe.rows)
        n1 = sc["starts"][-4]                          # start of the disturbance part
        dist = rows[rows[:, 0] >= n1]
        return {"seed": seed, "config": name,
                "ss_error_hold": [m["segments"][i]["ss_error"] for i in hold],
                "wrong_sign_share_all": float(np.mean(rows[:, 1] * rows[:, 2] <= 0)),
                "wrong_sign_share_disturbance_part": float(np.mean(dist[:, 1] * dist[:, 2] <= 0)),
                "gain_ratio_end": float(rows[-1, 1] / rows[-1, 2]),
                "window_tc_std_start": float(rows[0, 3]), "window_tc_std_end": float(rows[-1, 3]),
                "iae_total": m["iae_total"], "tv": m["tc_total_variation"], "violations": m["violations"],
                "refits": getattr(ident, "n_refit", None), "refits_rejected": getattr(ident, "n_refit_rejected", None),
                "probes": rows.tolist()}


def main():
    with ProcessPoolExecutor(rm.WORKERS) as ex:
        out = [r for rows in ex.map(run_seed, SEEDS) for r in rows]
    res = {"kind": "diagnostic by the truth (P3)", "scenario": "S0", "probe_every": PROBE_EVERY,
           "ss_limit": rm.SS_LIMIT, "runs": out}
    (rm.OUT / OUT_NAME).write_text(json.dumps(res, indent=1))
    for name in CONFIGS:
        rr = [r for r in out if r["config"] == name]
        bad = [r["seed"] for r in rr if max(abs(e) for e in r["ss_error_hold"]) > rm.SS_LIMIT]
        print(f"{name}: seeds over the limit {bad}; wrong-sign share in the disturbance part, median "
              f"{np.median([r['wrong_sign_share_disturbance_part'] for r in rr]):.3f}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
