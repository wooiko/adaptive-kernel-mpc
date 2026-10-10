"""Ignition protection by a hard upper limit of the coolant temperature (decision R2, variant B).

The cold steady state of the reactor disappears at the saddle-node point Tc* (cstr.saddle_node);
above it the reactor ignites. Tc* depends on the process parameters, which drift and are not
measured. The limit covers a named set of parameter deviations (the admissible set):
    UA  -5 .. 0 %   (fouling of the heat-transfer surface)
    Tf  -1 .. +1 K  (feed temperature)
    Tc_max = min over the vertices of the set of Tc*(p) - MARGIN.
The minimum over the vertices is checked against a grid over the whole set (tests/test_mpc.py):
no interior point gives a lower limit. Outside the set there is no protection.
Protection does not use the filtered measurement: it is a bound on the input, applied by every
controller and by the fallback plan.
"""
import itertools
from dataclasses import replace

import numpy as np

import cstr

TC_MIN = 292.0       # K: lower edge of the commissioning data (part 1, report section 6); below it the model extrapolates
MARGIN = 0.5         # K: margin below the worst saddle-node point (decision R2)
UA_RANGE = (0.95, 1.0)   # multiplier of the nominal UA (decision R2, variant B)
TF_RANGE = (-1.0, 1.0)   # K, added to the nominal Tf (decision R2, variant B)


def admissible_set(p0=cstr.CSTRParams()):
    """Parameter ranges covered by the protection, absolute values: {name: (lo, hi)}."""
    return {"UA": (UA_RANGE[0] * p0.UA, UA_RANGE[1] * p0.UA), "Tf": (p0.Tf + TF_RANGE[0], p0.Tf + TF_RANGE[1])}


def params_at(point, p0=cstr.CSTRParams()):
    """Process parameters with the values of `point` ({name: value}) substituted."""
    return replace(p0, **point)


def vertices(ranges):
    """All corners of the box `ranges` as {name: value} dicts."""
    names = list(ranges)
    return [dict(zip(names, c)) for c in itertools.product(*(ranges[n] for n in names))]


def limit_at(point, p0=cstr.CSTRParams()):
    """Saddle-node coolant temperature Tc* at one parameter point; ValueError if the cold and hot
    branches do not overlap there (no fold: the limit is not defined)."""
    Tc = cstr.saddle_node(params_at(point, p0))[0]
    if not np.isfinite(Tc):
        raise ValueError(f"no saddle-node point at {point}: no multiplicity, the limit is not defined")
    return float(Tc)


def tc_max(ranges=None, p0=cstr.CSTRParams(), margin=MARGIN):
    """Hard upper limit of Tc: the lowest saddle-node point over the vertices of the admissible set
    minus the margin. Also returns the worst vertex, its limit and Ca/T at the fold there, and Ca
    of the nominal cold steady state at Tc_max."""
    ranges = admissible_set(p0) if ranges is None else ranges
    lims = [(limit_at(v, p0), v) for v in vertices(ranges)]
    worst, v = min(lims, key=lambda t: t[0])
    _, ca_fold, t_fold = cstr.saddle_node(params_at(v, p0))
    out = worst - margin
    return {"tc_max": out, "worst_limit": worst, "worst_vertex": v, "margin": margin,
            "ca_at_fold": float(ca_fold), "T_at_fold": float(t_fold), "ranges": ranges,
            "vertex_limits": [(dict(vv), float(t)) for t, vv in lims],
            "ca_nominal_at_tc_max": cold_steady_state(out, p0)[0]}


def cold_steady_state(Tc, p=cstr.CSTRParams()):
    """(Ca, T) of the coldest steady state at a constant Tc."""
    ca, T = cstr.steady_states(Tc, p)[0]
    return float(ca), float(T)


def reachable_setpoints(u_min=TC_MIN, u_max=None, p=cstr.CSTRParams()):
    """Range of Ca reachable at steady state on the cold branch with u_min <= Tc <= u_max:
    (Ca at u_max, Ca at u_min); Ca falls monotonically as Tc rises on that branch."""
    u_max = tc_max(p0=p)["tc_max"] if u_max is None else u_max
    return cold_steady_state(u_max, p)[0], cold_steady_state(u_min, p)[0]
