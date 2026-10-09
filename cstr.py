"""Continuous stirred-tank reactor (CSTR) with an exothermic first-order reaction A -> B.

Textbook benchmark; parameter values follow Henson & Seborg, "Nonlinear Process
Control" (1997), as also used in the Seborg et al. process-control textbook.

States:  Ca [mol/L]  concentration of A in the reactor
         T  [K]      reactor temperature
Input:   Tc [K]      coolant temperature
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CSTRParams:
    q: float = 100.0       # feed flow rate, L/min
    V: float = 100.0       # reactor volume, L
    rho: float = 1000.0    # density, g/L
    Cp: float = 0.239      # heat capacity, J/(g K)
    mdelH: float = 5.0e4   # heat of reaction (-dH), J/mol
    EoverR: float = 8750.0 # activation energy / gas constant, K
    k0: float = 7.2e10     # pre-exponential factor, 1/min
    UA: float = 5.0e4      # heat-transfer coefficient * area, J/(min K)
    Caf: float = 1.0       # feed concentration, mol/L
    Tf: float = 350.0      # feed temperature, K


def rhs(t, x, Tc, p: CSTRParams):
    Ca, T = x
    r = p.k0 * np.exp(-p.EoverR / T) * Ca
    dCa = p.q / p.V * (p.Caf - Ca) - r
    dT = (p.q / p.V * (p.Tf - T)
          + p.mdelH / (p.rho * p.Cp) * r
          + p.UA / (p.V * p.rho * p.Cp) * (Tc - T))
    return [dCa, dT]


def steady_state(Tc, p: CSTRParams = CSTRParams(), x0=(0.9, 320.0)):
    """Integrate long enough to settle on the steady state reached from x0."""
    sol = solve_ivp(rhs, (0.0, 200.0), x0, args=(Tc, p), rtol=1e-9, atol=1e-11)
    return sol.y[:, -1]


def simulate(Tc_seq, Ts, x0, p: CSTRParams = CSTRParams(), k0_scale=None):
    """Simulate with a piecewise-constant input. Returns states at k = 0..N-1,
    where x[k] is the state at the moment Tc_seq[k] is applied.
    k0_scale (optional, one value per step) multiplies the reaction-rate constant:
    a simple way to imitate catalyst deactivation."""
    x = np.empty((len(Tc_seq), 2))
    xk = np.asarray(x0, dtype=float)
    for k, Tc in enumerate(Tc_seq):
        x[k] = xk
        pk = p if k0_scale is None else replace(p, k0=p.k0 * k0_scale[k])
        sol = solve_ivp(rhs, (0.0, Ts), xk, args=(Tc, pk), rtol=1e-8, atol=1e-10)
        xk = sol.y[:, -1]
    return x


def _T_balance(T, Tc, p):
    """Energy balance at a steady state, with Ca eliminated through the mass balance."""
    k = p.k0 * np.exp(-p.EoverR / T)
    Ca = p.q / p.V * p.Caf / (p.q / p.V + k)
    return rhs(0.0, [Ca, T], Tc, p)[1], Ca


def steady_states(Tc, p: CSTRParams = CSTRParams(), lo=300.0, hi=500.0, n=20001):
    """All steady states (Ca, T) at a given coolant temperature, ordered by T."""
    from scipy.optimize import brentq
    T = np.linspace(lo, hi, n)
    f = np.array([_T_balance(t, Tc, p)[0] for t in T])
    roots = [brentq(lambda t: _T_balance(t, Tc, p)[0], T[i], T[i + 1])
             for i in np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]]
    return [(_T_balance(t, Tc, p)[1], t) for t in roots]


def jacobian_eigenvalues(x, Tc, p: CSTRParams = CSTRParams(), h=1e-6):
    x = np.asarray(x, float)
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h * max(1.0, abs(x[j]))
        J[:, j] = (np.array(rhs(0, x + e, Tc, p)) - np.array(rhs(0, x - e, Tc, p))) / (2 * e[j])
    return np.linalg.eigvals(J)


def tc_of_T(T, p: CSTRParams = CSTRParams()):
    """Steady states, read the other way round. The energy balance is linear in Tc, so
    every steady state lies on the curve
        Tc = T - [q/V (Tf - T) + J r(T)] / b,   J = -dH/(rho Cp),  b = UA/(V rho Cp),
    with r(T) = k(T) Ca(T) and Ca(T) = (q/V) Caf / (q/V + k(T)) from the mass balance.
    Returns (Tc, Ca, dTc/dT); dr/dT = (q/V)^2 Caf k E/R / (T^2 (q/V + k)^2)."""
    a, J, b = p.q / p.V, p.mdelH / (p.rho * p.Cp), p.UA / (p.V * p.rho * p.Cp)
    k = p.k0 * np.exp(-p.EoverR / T)
    Ca = a * p.Caf / (a + k)
    Tc = T - (a * (p.Tf - T) + J * k * Ca) / b
    dr = a ** 2 * p.Caf * k * p.EoverR / (T ** 2 * (a + k) ** 2)
    return Tc, Ca, 1.0 + (a - J * dr) / b


def saddle_node(p: CSTRParams = CSTRParams(), T_lo=250.0, T_hi=600.0, n=35001):
    """Point where the low-temperature (cold) steady state disappears: the first local
    maximum of Tc(T) along T, i.e. the root of dTc/dT where it changes sign from + to -.
    Equivalent to f(T, Tc) = 0 and df/dT = 0 for the energy balance f with Ca eliminated.
    Returns (Tc, Ca, T); (nan, nan, nan) if the curve has no maximum (no multiplicity)."""
    from scipy.optimize import brentq
    T = np.linspace(T_lo, T_hi, n)
    d = tc_of_T(T, p)[2]
    i = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
    if len(i) == 0:
        return np.nan, np.nan, np.nan
    Ts = brentq(lambda t: tc_of_T(t, p)[2], T[i[0]], T[i[0] + 1], xtol=1e-12)
    Tc, Ca, _ = tc_of_T(Ts, p)
    return float(Tc), float(Ca), float(Ts)


def cold_branch_limit(p: CSTRParams = CSTRParams()):
    """Coolant temperature above which the cold steady state no longer exists: above it
    the reactor ignites."""
    return saddle_node(p)[0]
