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


def cold_branch_limit(p: CSTRParams = CSTRParams(), lo=300.0, hi=310.0, tol=1e-4):
    """Coolant temperature above which the low-temperature (cold) steady state no longer
    exists (saddle-node point): above it the reactor ignites. Bisection on the number of
    steady states in the cold part of the temperature axis."""
    has_cold = lambda Tc: len(steady_states(Tc, p, 300.0, 360.0, 60001)) >= 2
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if has_cold(m) else (lo, m)
    return lo
