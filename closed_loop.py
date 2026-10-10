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
from dataclasses import replace

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


def run_loop(plant, x0, ident, controller, r, add, missing, u0, supervisor=True):
    """Run len(r) samples. x0: initial state; u0: input held before the controller starts (samples 0, 1).
    ident: identifier (its `horizon` sets the length of u_plan); controller: step(k, y, u, r_k, info) and
    planned_inputs(u_prev, n). supervisor=False: the controller gets m_k = 1 (the regime is still logged).
    Returns the step log (arrays)."""
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
        info = {"mult": res["mult"] if supervisor else 1.0, "missing": bool(np.isnan(y_meas[k])), "rejected": bool(res["flagged"]),
                "x_true": X[k], "params": plant.params(k) if hasattr(plant, "params") else None}   # oracle only
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


def _with(p0=cstr.CSTRParams(), UA=1.0, Tf=0.0, k0=1.0, Caf=1.0):
    """Process parameters with relative changes (UA, k0, Caf: multipliers; Tf: added, K)."""
    return replace(p0, UA=p0.UA * UA, Tf=p0.Tf + Tf, k0=p0.k0 * k0, Caf=p0.Caf * Caf)


def _ramp(k, k1, k2, v1, v2):
    """Linear change from v1 at sample k1 to v2 at sample k2, constant outside."""
    return v1 + (v2 - v1) * min(max((k - k1) / (k2 - k1), 0.0), 1.0)


def _build(name, seed, r, params, x0, u0, starts, gaps=(), series=(), **extra):
    """Scenario with its own random stream (data.streams(seed, SCENARIO_IDS[name]))."""
    n = len(r)
    _, noise_seed = data.streams(seed, SCENARIO_IDS[name])
    add, spike, missing = measurement_noise(n, noise_seed, gaps=gaps)
    for s0, length, size in series:                        # consecutive spikes of `size` noise std
        add[s0:s0 + length] += size * data.NOISE_STD
        spike[s0:s0 + length] = True
    return {"name": name, "plant": CSTRPlant(params), "x0": np.asarray(x0, float), "u0": float(u0),
            "r": np.asarray(r, float), "add": add, "spike": spike, "missing": missing, "starts": list(starts),
            **extra}


S2_TC = 297.0                # K: set-point of S2 = nominal Ca at this Tc (reachable for every S2 disturbance)
S2_EVENTS = (100, 250, 400, 550, 850)   # Tf +1 K, Tf -1 K, Tf back, start and end of the UA ramp to -5 %
S2_N = 1000


def scenario_s2(seed, tc_max):
    """S2: constant set-point; steps of Tf by +-1 K, then UA falls by 5 % over 30 min (TZ 8)."""
    e = S2_EVENTS

    def params(k):
        tf = 1.0 if e[0] <= k < e[1] else (-1.0 if e[1] <= k < e[2] else 0.0)
        return _with(UA=_ramp(k, e[3], e[4], 1.0, 0.95), Tf=tf)
    ca = safety.cold_steady_state(S2_TC)[0]
    return _build("S2", seed, np.full(S2_N, ca), params, safety.cold_steady_state(S2_TC), S2_TC, e,
                  hold_segments=[0, 1, 2, 4])      # segments with a constant disturbance at their end


S3_HOLD, S3_STEPS, S3_K0_END = 300, 10, 0.8   # 30 min per set-point, 300 min of drift, k0 x0.8 at the end
S3_LEVELS = (0.5, 0.15, 0.85, 0.3, 0.7, 0.05, 0.95, 0.4, 0.6, 0.2)   # shares of the common reachable range


def s3_range(tc_max):
    """Ca reachable at steady state for every k0 multiplier in [S3_K0_END, 1]: Ca is monotone in k0, so the
    common range is [Ca(tc_max) at k0 = S3_K0_END, Ca(TC_MIN) at nominal k0]."""
    lo = safety.cold_steady_state(tc_max, _with(k0=S3_K0_END))[0]
    hi = safety.cold_steady_state(safety.TC_MIN)[0]
    return lo, hi


def scenario_s3(seed, tc_max):
    """S3: k0 falls linearly to x0.8 over 300 min, set-point changed every 30 min inside the range reachable
    throughout (a lower k0 only raises the ignition limit, part 1 table)."""
    lo, hi = s3_range(tc_max)
    margin = 0.1 * (hi - lo)
    levels = [lo + margin + f * (hi - lo - 2 * margin) for f in S3_LEVELS]
    x0 = safety.cold_steady_state(296.0)
    r = np.r_[np.full(S1_INIT, x0[0]), np.repeat(levels, S3_HOLD)]
    params = lambda k: _with(k0=_ramp(k, S1_INIT, len(r), 1.0, S3_K0_END))
    return _build("S3", seed, r, params, x0, 296.0, [S1_INIT + S3_HOLD * i for i in range(S3_STEPS)])


S4_TC, S4_HOLD = (296.0, 299.0, 294.0), 150
S4_GAP = (200, 8)                 # logger gap: start, samples (as in part 1)
S4_SERIES = (380, 3, 15.0)        # three consecutive spikes of 15 noise std (the filter replaces up to 3 in a row)


def scenario_s4(seed, tc_max, spikes=True):
    """S4: measurement faults (2 % spikes, a logger gap, a run of spikes) on set-point steps; spikes=False
    gives the reference run with the same noise and gap but no spikes."""
    ca = [safety.cold_steady_state(t)[0] for t in S4_TC]
    r = np.r_[np.full(S1_INIT, ca[0]), np.repeat(ca[1:], S4_HOLD), np.full(S4_HOLD, ca[0])]
    sc = _build("S4", seed, r, lambda k: cstr.CSTRParams(), safety.cold_steady_state(S4_TC[0]), S4_TC[0],
                [S1_INIT, S1_INIT + S4_HOLD, S1_INIT + 2 * S4_HOLD], gaps=[S4_GAP],
                series=[S4_SERIES] if spikes else ())
    if not spikes:
        sc["add"] = sc["add"] - _spike_part(seed, len(r))
        sc["spike"][:] = False
    return sc


def _spike_part(seed, n):
    """The spike component of the S4 measurement error (same stream), to build the no-spike reference."""
    _, noise_seed = data.streams(seed, SCENARIO_IDS["S4"])
    with_spikes = measurement_noise(n, noise_seed)[0]
    without = measurement_noise(n, noise_seed, anomaly_rate=0.0)[0]
    return with_spikes - without


S5_EVENTS = (50, 150, 450, 750)   # set-point to the edge; Tf +1 K and start of the UA ramp; ramp end; end


def scenario_s5(seed, tc_max, outside=False):
    """S5a: the plant moves to the worst vertex of the admissible set (Tf +1 K, UA -5 % over 30 min) with
    the set-point at the edge of the range: the cold Ca at that vertex at Tc_max. S5b (outside=True): the
    same set-point, UA -10 % and Caf +5 %, outside the set: no protection is claimed there."""
    e = S5_EVENTS
    worst = safety.tc_max()["worst_vertex"]
    p0 = cstr.CSTRParams()
    edge = safety.cold_steady_state(tc_max, safety.params_at(worst))[0]
    x0 = safety.cold_steady_state(296.0)
    r = np.r_[np.full(e[0], x0[0]), np.full(e[3] - e[0], edge)]
    if outside:
        params = lambda k: _with(UA=_ramp(k, e[1], e[2], 1.0, 0.90), Caf=1.05 if k >= e[1] else 1.0)
    else:
        params = lambda k: _with(UA=_ramp(k, e[1], e[2], 1.0, worst["UA"] / p0.UA),
                                 Tf=(worst["Tf"] - p0.Tf) if k >= e[1] else 0.0)
    return _build("S5b" if outside else "S5a", seed, r, params, x0, 296.0, [e[0], e[1], e[2]])


S6_HOLD = (150, 100, 150, 100)    # above the range, back inside, below the range, back inside
S6_OUT = 0.01                     # mol/L beyond the reachable range


def scenario_s6(seed, tc_max):
    """S6: set-point outside the reachable range on both sides, each followed by a return inside."""
    hi = safety.cold_steady_state(safety.TC_MIN)[0] + S6_OUT
    lo = safety.cold_steady_state(tc_max)[0] - S6_OUT
    mid = safety.cold_steady_state(296.0)[0]
    r = np.r_[np.full(S1_INIT, mid), np.full(S6_HOLD[0], hi), np.full(S6_HOLD[1], mid),
              np.full(S6_HOLD[2], lo), np.full(S6_HOLD[3], mid)]
    starts = [int(v) for v in S1_INIT + np.cumsum((0,) + S6_HOLD[:-1])]
    return _build("S6", seed, r, lambda k: cstr.CSTRParams(), safety.cold_steady_state(296.0), 296.0, starts)


SCENARIOS = {"S0": scenario_s0, "S1": scenario_s1, "S2": scenario_s2, "S3": scenario_s3, "S4": scenario_s4,
             "S5a": scenario_s5, "S5b": lambda seed, lim: scenario_s5(seed, lim, outside=True), "S6": scenario_s6}


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
