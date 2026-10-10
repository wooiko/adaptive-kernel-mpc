"""Closed loop, one sample at a time: plant, measurement, identifier, controller; scenarios and metrics.

Order of events at sample k (time convention of TZ 4 and mpc.py):
  1. y_meas[k] arrives: the true Ca at the start of step k plus sensor noise, spikes or a gap. The noise,
     spikes and gaps of a run are drawn before it starts, so every controller sees the same realisation.
  2. Identifier step on (x_k-1, y_meas[k]) (adaptive.AdaptiveIdentifier.step). The supervisor checks
     the controller's planned inputs u_k .. u_k+H-2 (its last plan shifted, TZ 6.2). The filter decision
     gives y_rt[k], which enters the history; a run of rejections recognised as a process change puts
     the measurements back (the same handling as adaptive.run_online).
  3. Controller step: u_k from the history as it stands at sample k (y_rt[k] and earlier values, a
     restored measurement included: it is known at k) and the supervisor's multiplier m_k. Nothing
     from later samples is used (test: future measurements replaced by NaN leave u[0..k] unchanged).
  4. The plant integrates over [t_k, t_k+1) with u_k held, giving the state at k+1.
The set-point is not previewed: the controller uses r[k] over its whole horizon.
Metrics are computed on the true Ca of the simulator; values on the measurement are only additional.
"""
import time

import numpy as np
from scipy.integrate import solve_ivp

import cstr
import data
import safety
from adaptive import LAG, regressor

IGNITION_T = 350.0   # K. On the cold branch T stays below it at every vertex of the admissible set even at
                     # the limit Tc_max, and the hot steady state just above the nominal limit is far above
                     # it (both computed in tests/test_mpc.py, class IgnitionThreshold).
BAND = 0.005         # mol/L, settling band (TZ 8)
SS_WINDOW = 50       # samples (5 min): steady-state error = mean over the last 5 min of a segment (TZ 8)
SCENARIO_IDS = {"S0": 10, "S1": 11, "S2": 12, "S3": 13, "S4": 14, "S5a": 15, "S5b": 16, "S6": 17}  # >= 10 (P5)
S1_HOLD, S1_INIT = 100, 50   # samples: 10 min per set-point (TZ 8), 5 min on the initial one
S1_BELOW_LIMIT = 0.25        # K: the set-point near the limit is the nominal Ca at Tc_max - 0.25 K


class CSTRPlant:
    """The reactor of cstr.py; params(k) gives the process parameters on step k (drift, faults)."""
    def __init__(self, params=None):
        self.params = params or (lambda k: cstr.CSTRParams())

    def step(self, x, u, k):
        sol = solve_ivp(cstr.rhs, (0.0, data.TS), x, args=(u, self.params(k)), rtol=1e-8, atol=1e-10)
        return sol.y[:, -1]

    @staticmethod
    def ca(x): return float(x[0])

    @staticmethod
    def temp(x): return float(x[1])


class LinearPlant:
    """Test plant y_k+1 = theta . [y_k, y_k-1, u_k, u_k-1]; state [y_k, y_k-1, u_k-1]."""
    def __init__(self, theta):
        self.theta = np.asarray(theta, float)

    def step(self, x, u, k):
        return np.array([self.theta @ [x[0], x[1], u, x[2]], x[0], u])

    @staticmethod
    def ca(x): return float(x[0])

    @staticmethod
    def temp(x): return np.nan


def measurement_noise(n, noise_seed, anomaly_rate=data.ANOMALY_RATE, gaps=()):
    """Additive sensor error for n samples, built as data.make_log does: Gaussian noise, isolated spikes of
    5-25 noise std in a share `anomaly_rate` of the samples (not in the first 5), and gaps (start, length).
    Returns (add, is_spike, missing)."""
    rng = np.random.default_rng(noise_seed)
    add = rng.normal(0.0, data.NOISE_STD, n)
    spike = np.zeros(n, bool)
    idx = rng.choice(np.arange(5, n), size=int(anomaly_rate * n), replace=False)
    add[idx] += rng.choice([-1, 1], len(idx)) * rng.uniform(5, 25, len(idx)) * data.NOISE_STD
    spike[idx] = True
    missing = np.zeros(n, bool)
    for s, length in gaps:
        missing[s:s + length] = True
    spike &= ~missing
    return add, spike, missing


def run_loop(plant, x0, ident, controller, r, add, missing, u0):
    """Run len(r) samples. x0: initial state; u0: input held before the controller starts (samples 0, 1).
    ident: identifier (its `horizon` sets the length of u_plan); controller: step(k, y, u, r_k, info) and
    planned_inputs(u_prev, n). Returns the step log (arrays)."""
    n = len(r)
    H = getattr(ident, "horizon", 0)
    X = np.empty((n, len(x0)))
    u = np.full(n, np.nan)
    u[:LAG] = u0
    y_meas, y_hist = np.full(n, np.nan), np.full(n, np.nan)
    log = {key: np.full(n, np.nan) for key in ("y_rt", "y_ctrl", "mult", "var_h", "d", "y_pred_next",
                                               "t_ctrl_ms", "t_ident_ms")}
    log.update({"flagged_rt": np.zeros(n, bool), "regime": np.zeros(n, int), "fallback": np.zeros(n, bool),
                "restored": np.zeros(n, bool), "status": [""] * n})
    X[0] = x0
    for k in range(n):
        if k > 0:
            X[k] = plant.step(X[k - 1], u[k - 1], k - 1)
        y_meas[k] = np.nan if missing[k] else plant.ca(X[k]) + add[k]
        if k < LAG:
            y_hist[k] = log["y_rt"][k] = y_meas[k] if np.isfinite(y_meas[k]) else plant.ca(x0)
            continue
        u_plan = controller.planned_inputs(u[k - 1], H - 1) if H else None
        t0 = time.perf_counter()
        res = ident.step(regressor(y_hist, u, k - 1), y_meas[k], u_plan)
        y_hist[k] = log["y_rt"][k] = res["y_keep"]
        if res["restore"]:
            idx = k - np.array(res["restore"])
            y_hist[idx] = y_meas[idx]
            log["restored"][idx] = True
            for j in range(idx.min() + 1, k):           # a gap inside the run: predict it again
                if np.isnan(y_meas[j]):
                    y_hist[j] = ident.predict(regressor(y_hist, u, j - 1))
            ident.learn(np.array([regressor(y_hist, u, j - 1) for j in idx]), y_meas[idx], res["restore"])
        log["t_ident_ms"][k] = 1e3 * (time.perf_counter() - t0)
        info = {"mult": res["mult"], "missing": bool(np.isnan(y_meas[k])), "rejected": bool(res["flagged"])}
        t0 = time.perf_counter()
        u[k], det = controller.step(k, y_hist, u, r[k], info)
        log["t_ctrl_ms"][k] = 1e3 * (time.perf_counter() - t0)
        log["y_ctrl"][k] = det.get("y_used", np.nan)     # the y_k the controller actually used
        log["flagged_rt"][k], log["regime"][k] = res["flagged"], res["regime"]
        log["mult"][k], log["var_h"][k] = res["mult"], res["var_h"]
        for key in ("d", "y_pred_next", "fallback", "status"):
            log[key][k] = det.get(key, log[key][k])
    log.update({"Ca": np.array([plant.ca(x) for x in X]), "T": np.array([plant.temp(x) for x in X]),
                "u": u, "y_meas": y_meas, "r": np.asarray(r, float), "y_hist_end": y_hist})
    return log


# ---------------------------------------------------------------- scenarios

def s1_setpoints(tc_max, p=cstr.CSTRParams()):
    """S1: initial set-point and six steps through the whole reachable range, one of them near the
    limit. The set-points are nominal steady-state Ca at chosen coolant temperatures (fixed design,
    not tuned on any result)."""
    top = tc_max - S1_BELOW_LIMIT
    tcs = (296.0, top, 293.0, 298.5, safety.TC_MIN + 0.5, top - 0.5, 295.0)
    return np.array(tcs), np.array([safety.cold_steady_state(t, p)[0] for t in tcs])


def scenario_s1(seed, tc_max):
    tcs, ca = s1_setpoints(tc_max)
    n = S1_INIT + S1_HOLD * (len(ca) - 1)
    r = np.r_[np.full(S1_INIT, ca[0]), np.repeat(ca[1:], S1_HOLD)]
    _, noise_seed = data.streams(seed, SCENARIO_IDS["S1"])
    add, spike, missing = measurement_noise(n, noise_seed)
    x0 = np.array(safety.cold_steady_state(tcs[0]))
    starts = [S1_INIT + S1_HOLD * i for i in range(len(ca) - 1)]
    return {"name": "S1", "plant": CSTRPlant(), "x0": x0, "u0": tcs[0], "r": r, "add": add, "spike": spike,
            "missing": missing, "starts": starts, "tc_levels": tcs}


S0_STEPS = 8                 # S0 (tuning only): eight set-point steps of S1_HOLD samples


def scenario_s0(seed, tc_max):
    """S0, tuning only (rho, PID settings, sensitivity to Np): its own random stream, set-points drawn
    as nominal steady-state Ca at coolant temperatures uniform in [TC_MIN + 0.5, tc_max - S1_BELOW_LIMIT]."""
    rng, noise_seed = data.streams(seed, SCENARIO_IDS["S0"])
    tcs = rng.uniform(safety.TC_MIN + 0.5, tc_max - S1_BELOW_LIMIT, S0_STEPS + 1)
    ca = np.array([safety.cold_steady_state(t)[0] for t in tcs])
    n = S1_INIT + S1_HOLD * S0_STEPS
    r = np.r_[np.full(S1_INIT, ca[0]), np.repeat(ca[1:], S1_HOLD)]
    add, spike, missing = measurement_noise(n, noise_seed)
    starts = [S1_INIT + S1_HOLD * i for i in range(S0_STEPS)]
    return {"name": "S0", "plant": CSTRPlant(), "x0": np.array(safety.cold_steady_state(tcs[0])), "u0": tcs[0],
            "r": r, "add": add, "spike": spike, "missing": missing, "starts": starts, "tc_levels": tcs}


# ---------------------------------------------------------------- metrics

def settling_time(ca, r, s, e, band=BAND):
    """Samples from the event s until |ca - r| stays within the band up to the end e of the segment;
    the search starts at s, never before (P2). None if it never settles."""
    out = np.abs(ca[s:e] - r[s:e]) > band
    if out[-1]:
        return None
    last_out = np.where(out)[0]
    return int(last_out[-1] + 1) if len(last_out) else 0


def segment_metrics(ca, r, starts, n, ts=data.TS, band=BAND, ss_window=SS_WINDOW):
    """Per set-point segment [s, next start): IAE (mol/L min), settling time (min), overshoot past the new
    set-point in the direction of the step (mol/L and % of the step), steady-state error (mean of
    ca - r over the last ss_window samples, mol/L)."""
    out = []
    ends = list(starts[1:]) + [n]
    for s, e in zip(starts, ends):
        step = r[s] - r[s - 1]
        over = float(max(np.max(np.sign(step) * (ca[s:e] - r[s])), 0.0)) if step != 0 else 0.0
        st = settling_time(ca, r, s, e, band)
        out.append({"start": int(s), "setpoint": float(r[s]), "step": float(step),
                    "iae": float(np.sum(np.abs(ca[s:e] - r[s:e])) * ts),
                    "settling_min": None if st is None else st * ts,
                    "overshoot": over, "overshoot_pct": 100.0 * over / abs(step) if step != 0 else 0.0,
                    "ss_error": float(np.mean(ca[e - ss_window:e] - r[e - ss_window:e]))})
    return out


def run_metrics(log, starts, u_min, u_max, du_max, k0=LAG):
    """Whole-run metrics on the true Ca; k0: first controlled sample."""
    ca, r, u, T = log["Ca"], log["r"], log["u"], log["T"]
    n = len(r)
    du = np.diff(u)
    tc = log["t_ctrl_ms"][k0:]
    ti = log["t_ident_ms"][k0:]
    tol = 1e-9
    return {"segments": segment_metrics(ca, r, starts, n),
            "iae_total": float(np.sum(np.abs(ca[k0:] - r[k0:])) * data.TS),
            "iae_total_on_measurement": float(np.nansum(np.abs(log["y_meas"][k0:] - r[k0:])) * data.TS),
            "tc_total_variation": float(np.sum(np.abs(du))),
            "share_at_upper_limit": float(np.mean(np.abs(u[k0:] - u_max) < 1e-6)),
            "share_at_lower_limit": float(np.mean(np.abs(u[k0:] - u_min) < 1e-6)),
            "max_T": float(np.nanmax(T)), "ignition": bool(np.nanmax(T) > IGNITION_T),
            "violations": int(np.sum(u > u_max + tol) + np.sum(u < u_min - tol) + np.sum(np.abs(du) > du_max + tol)),
            "fallback_steps": int(np.sum(log["fallback"][k0:])), "n_steps": int(n - k0),
            "statuses": {s: int(sum(1 for x in log["status"][k0:] if x == s)) for s in set(log["status"][k0:])},
            "step_ms": {"median": float(np.median(tc)), "p95": float(np.percentile(tc, 95)), "max": float(np.max(tc))},
            "ident_ms": {"median": float(np.median(ti)), "p95": float(np.percentile(ti, 95)), "max": float(np.max(ti))},
            "share_not_normal": float(np.mean(log["regime"][k0:] > 0)),
            "n_flagged_rt": int(np.sum(log["flagged_rt"]))}
