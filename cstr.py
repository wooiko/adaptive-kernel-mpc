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
