"""K-MPC: predictive control of Ca through the coolant temperature Tc on a kernel NARX model,
linearised along a nominal trajectory at every step and solved as a convex QP (OSQP).

Time convention (the same as cstr.simulate and adaptive.regressor). At sample k the measurement
y_k (state at the start of the step) arrives; the controller computes u_k, which is held on
[t_k, t_k+1). There is no direct feed-through: u_k first affects y_{k+1}, and the regressor
x_k = [y_k, y_{k-1}, u_k, u_{k-1}] contains the decision u_k.

Model, physical units:  y_{j+1} = f(x_j),   g(x) = df/dx = [a1, a2, b1, b2]  (adaptive.regressor order).

1. Nominal trajectory. Ubar = previous optimal plan shifted by one step, last value repeated
   (u_{k-1} over the whole horizon on the first step; `shift_plan`). Free run of the model from
   the last real-time measurements (the filter's y_rt, never the history restored later):
       ybar_{k+i+1} = f(xbar_{k+i}) + d_k,   xbar_{k+i} = [ybar_{k+i}, ybar_{k+i-1}, ubar_{k+i}, ubar_{k+i-1}],
       ybar_k = y_k,  ybar_{k-1} = y_{k-1},  ubar_{k-1} = u_{k-1},  i = 0 .. Np-1,
   with the disturbance estimate d_k: the one-step prediction error y_k - f(x_{k-1}) (`disturbance`),
   passed through a first-order filter with gain beta in KMPC (accepted 10.10.2026, see KMPC).
   d_k enters every step of the recursion (variant A, accepted 10.10.2026), not added after it
   as written in TZ 5.4: the free run starts from y_k, which already contains the disturbance,
   and for an autoregressive model a constant added after the recursion leaves a steady-state
   error. With d_k inside, an output disturbance on an exactly known linear plant is predicted
   exactly, and a constant disturbance gives zero steady-state error (tests/test_mpc.py).
2. Linearisation and sensitivity. The coefficients a1, a2, b1, b2 are taken at every nominal point
   (time-varying linearisation; `linearize`). Small deviations obey
       dy_{k+i+1} = a1_i dy_{k+i} + a2_i dy_{k+i-1} + b1_i du_{k+i} + b2_i du_{k+i-1},
       dy_k = dy_{k-1} = 0,  du_{k-1} = 0,
   so dY = G dU, G lower triangular Np x Np, column j = response to a unit du_{k+j}
   (`sensitivity`); d_k is fixed during the step and does not enter G. The input is held after
   Nc moves, U = M V (`blocking_matrix`), Phi = G M, and Ubar = M Vbar. G is the exact Jacobian of
   the free run with respect to the input at Ubar.
       Yhat = Ybar + Phi (V - Vbar) = c + Phi V,   c = Ybar - Phi Vbar.
3. QP (TZ 5.5). Variable V = [u_k .. u_{k+Nc-1}], K. Moves dU = D V - e u_{k-1}, D: 1 on the
   diagonal, -1 below it; e = [1, 0, .., 0]. With Q = q0 I and the move weight W:
       J(V) = (c + Phi V - r)' Q (c + Phi V - r) + (D V - e u_{k-1})' W (D V - e u_{k-1})
   * symmetric W = m r0 I:  min 1/2 V'PV + q'V,
       P = 2 (Phi'Q Phi + D'W D),  q = 2 (Phi'Q (c - r) - D'W e u_{k-1});
   * asymmetric (TZ 6.2, the default): the supervisor's multiplier m acts only on heating moves.
     Variables z = [V, p, n], dU = p - n, p, n >= 0, move cost m r0 p'p + r0 n'n:
       P = blockdiag(2 Phi'Q Phi, 2 m r0 I, 2 r0 I),  q = [2 Phi'Q (c - r), 0, 0],
       equality rows  D V - p + n = e u_{k-1}.
     Both costs are strictly convex in (p, n); at the optimum p_j n_j = 0 (lowering both by the
     smaller one keeps p - n and lowers the cost), so the move cost is m r0 dU^2 for dU > 0 and
     r0 dU^2 for dU < 0. With m = 1 the two forms have the same solution.
   Hard constraints, always:  u_min <= V <= u_max,  -du_max <= D V - e u_{k-1} <= du_max.
   V = u_{k-1} 1 is feasible whenever u_min <= u_{k-1} <= u_max, so the QP has a solution.
4. Solution (`Solver`, `plan`). OSQP, warm-started from the shifted plan; the problem is solved in
   deviations w = V - u_{k-1} 1 (same solution, better scaled than kelvin). u_k = V_0 is applied;
   V is kept as the next nominal plan and as u_plan for the supervisor. A result is used only with
   status 'solved' and bound violations below FEAS_TOL (then clipped to the bounds); otherwise the
   shifted previous plan, clipped to the constraints, is applied (`clip_plan`) and the step is counted.
   n_iter >= 1 repeats the linearisation around the new plan.
The closed loop gives no stability guarantee; see the README for what the simulation shows.
"""
import os
import sys
from contextlib import contextmanager
from typing import Callable, NamedTuple

import numpy as np
import osqp
import scipy.sparse as sp

from adaptive import NA, NB
from safety import TC_MIN

assert NA == 2 and NB == 2, "the trajectory code below is written for x = [y_k, y_k-1, u_k, u_k-1]"

NP, NC = 20, 5           # prediction and control horizons, samples (decision R4)
DU_MAX = 1.0             # K per step, |Tc move| (decision R5: a design parameter, no physical source)
SIGMA_Y = 0.01           # mol/L, output scale of the weights: q0 = 1 / SIGMA_Y^2 (TZ 5.5)
SIGMA_U = 1.0            # K, input scale: r0 = rho / SIGMA_U^2; rho is tuned on S0, never on test data
OSQP_SETTINGS = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 20000, "polishing": True, "verbose": False}
FEAS_TOL = 1e-6          # K: largest bound violation accepted from a 'solved' result before clipping


def q0_default():
    return 1.0 / SIGMA_Y ** 2


def r0_of(rho):
    return rho / SIGMA_U ** 2


class Model(NamedTuple):
    """One-step model in physical units: f(x) -> y_{k+1}, g(x) -> df/dx, x = [y_k, y_k-1, u_k, u_k-1]."""
    f: Callable
    g: Callable


def window_model(ident):
    """The identifier's current sliding-window KRR (adaptive.AdaptiveIdentifier). The view is
    live: after a refit of the window the same object predicts with the new coefficients."""
    return Model(ident.predict, ident.gradient)


def static_model(m):
    """pipeline.StaticKRR fitted once at commissioning."""
    return Model(lambda x: float(m.predict(x)[0]), m.gradient)


def linear_model(theta, c0=0.0):
    """y_{k+1} = theta . x + c0 (e.g. pipeline.LinearOE: theta = p[:-1], c0 = p[-1])."""
    th = np.asarray(theta, float)
    return Model(lambda x: float(np.dot(th, x) + c0), lambda x: th.copy())


def disturbance(model, y_k, x_km1):
    """d_k = y_k - f(x_{k-1}): error of the one-step prediction made at k-1 for k, both from
    real-time data (y_rt)."""
    return float(y_k - model.f(np.asarray(x_km1, float)))


def shift_plan(V, u_km1, Nc=NC):
    """Nominal plan for this step: the previous plan V (u_{k-1} .. u_{k+Nc-2}) shifted by one step,
    its last value repeated; on the first step (V is None) u_{k-1} for every move."""
    if V is None:
        return np.full(Nc, float(u_km1))
    V = np.asarray(V, float)
    return np.r_[V[1:], V[-1]]


def blocking_matrix(Np=NP, Nc=NC):
    """M (Np x Nc): U = M V, u_{k+i} = v_min(i, Nc-1); the input is held after the Nc-th move."""
    M = np.zeros((Np, Nc))
    M[np.arange(Np), np.minimum(np.arange(Np), Nc - 1)] = 1.0
    return M


def nominal(model, y_k, y_km1, u_km1, Ubar, d=0.0):
    """Free run along Ubar (u_k .. u_{k+Np-1}) from y_k, y_{k-1}, u_{k-1}, with d inside the
    recursion. Returns Ybar = [ybar_{k+1} .. ybar_{k+Np}] and the regressors Xbar[i] = xbar_{k+i}."""
    Ubar = np.asarray(Ubar, float)
    Np = len(Ubar)
    Y, X = np.empty(Np), np.empty((Np, 4))
    y0, y1, up = float(y_k), float(y_km1), float(u_km1)
    for i in range(Np):
        X[i] = (y0, y1, Ubar[i], up)
        y_next = model.f(X[i]) + d
        Y[i] = y_next
        y0, y1, up = y_next, y0, Ubar[i]
    return Y, X


def linearize(model, Xbar):
    """Rows [a1, a2, b1, b2] = g(xbar_{k+i}) at every nominal regressor."""
    return np.array([model.g(x) for x in Xbar])


def sensitivity(coef):
    """G (Np x Np), G[i, j] = d ybar_{k+i+1} / d ubar_{k+j}, from the recursion in the module
    docstring: row i is the deviation of y_{k+i+1}; row i-1 is that of y_{k+i} (zero for i = 0),
    row i-2 that of y_{k+i-1} (zero for i < 2); du_{k+i} is column i, du_{k+i-1} column i-1."""
    coef = np.asarray(coef, float)
    Np = len(coef)
    G = np.zeros((Np, Np))
    for i in range(Np):
        a1, a2, b1, b2 = coef[i]
        if i >= 1:
            G[i] += a1 * G[i - 1]
            G[i, i - 1] += b2
        if i >= 2:
            G[i] += a2 * G[i - 2]
        G[i, i] += b1
    return G


def difference_matrix(Nc=NC):
    return np.eye(Nc) - np.eye(Nc, k=-1)


class QP(NamedTuple):
    """min 1/2 z'Pz + q'z  s.t.  l <= A z <= u.  z[:nv] = V; the rest (asymmetric form) = [p, n]."""
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
    nv: int
    u_km1: float


def build_qp(Phi, Ybar, Vbar, r, u_km1, q0, r0, mult=1.0, asymmetric=True,
             u_min=TC_MIN, u_max=np.inf, du_max=DU_MAX):
    """QP of step 3 in the module docstring. r: set-point, scalar or one value per horizon step.
    mult: the supervisor's multiplier m_k (1 .. 2), on heating moves only if asymmetric."""
    Phi, Ybar, Vbar = np.asarray(Phi, float), np.asarray(Ybar, float), np.asarray(Vbar, float)
    Np, Nc = Phi.shape
    r = np.broadcast_to(np.asarray(r, float), (Np,))
    c = Ybar - Phi @ Vbar
    D, I = difference_matrix(Nc), np.eye(Nc)
    e = np.zeros(Nc); e[0] = 1.0
    Hy = 2.0 * q0 * Phi.T @ Phi
    gy = 2.0 * q0 * Phi.T @ (c - r)
    box_l, box_u = np.full(Nc, u_min), np.full(Nc, u_max)
    rate_l, rate_u = -du_max + e * u_km1, du_max + e * u_km1
    if not asymmetric:
        W = r0 * mult * I
        P = Hy + 2.0 * D.T @ W @ D
        q = gy - 2.0 * D.T @ W @ e * u_km1
        A = np.vstack([I, D])
        return QP(P, q, A, np.r_[box_l, rate_l], np.r_[box_u, rate_u], Nc, float(u_km1))
    Z = np.zeros((Nc, Nc))
    P = np.block([[Hy, Z, Z], [Z, 2.0 * mult * r0 * I, Z], [Z, Z, 2.0 * r0 * I]])
    q = np.r_[gy, np.zeros(2 * Nc)]
    A = np.block([[I, Z, Z], [D, Z, Z], [D, -I, I], [Z, I, Z], [Z, Z, I]])
    l = np.r_[box_l, rate_l, e * u_km1, np.zeros(2 * Nc)]
    u = np.r_[box_u, rate_u, e * u_km1, np.full(2 * Nc, np.inf)]
    return QP(P, q, A, l, u, Nc, float(u_km1))


def clip_plan(V, u_km1, u_min=TC_MIN, u_max=np.inf, du_max=DU_MAX):
    """Plan clipped to the hard constraints, move by move: each value within du_max of the previous
    input, then within [u_min, u_max]. If both cannot hold (u_{k-1} farther than du_max outside the
    box, which no controller here produces) the box wins: it is the ignition protection."""
    out, prev = np.empty(len(V)), float(u_km1)
    for j, v in enumerate(np.asarray(V, float)):
        prev = out[j] = min(max(min(max(v, prev - du_max), prev + du_max), u_min), u_max)
    return out


@contextmanager
def _quiet_stdout():
    """OSQP 1.1.3 prints 'Polishing not needed ...' to the C-level stdout even with verbose=False
    (observed on Python 3.13); the descriptor is redirected to the null device while it solves."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


class Solver:
    """OSQP with a warm start from the shifted plan (one instance per controller)."""
    def __init__(self, settings=None):
        self.settings = dict(OSQP_SETTINGS if settings is None else settings)

    def solve(self, qp, z0=None):
        """Returns (z, status). Solved in deviations w = z - s, s = u_{k-1} on the V part."""
        n = len(qp.q)
        s = np.zeros(n); s[:qp.nv] = qp.u_km1
        As = qp.A @ s
        m = osqp.OSQP()
        m.setup(sp.triu(sp.csc_matrix(qp.P), format="csc"), qp.q + qp.P @ s, sp.csc_matrix(qp.A),
                qp.l - As, qp.u - As, **self.settings)
        if z0 is not None:
            m.warm_start(x=np.asarray(z0, float) - s)
        with _quiet_stdout():
            res = m.solve(raise_error=False)
        status = str(res.info.status)
        if status != "solved" or res.x is None or not np.all(np.isfinite(res.x)):
            return None, status
        z = res.x + s
        Az = qp.A @ z
        if np.max(np.r_[qp.l - Az, Az - qp.u, 0.0]) > FEAS_TOL:
            return None, "bound violation"
        return z, status


def warm_start_point(Vbar, u_km1, asymmetric):
    """Primal warm start: the shifted plan, with its moves split into p and n."""
    if not asymmetric:
        return np.asarray(Vbar, float)
    dU = difference_matrix(len(Vbar)) @ Vbar - np.r_[u_km1, np.zeros(len(Vbar) - 1)]
    return np.r_[Vbar, np.maximum(dU, 0.0), np.maximum(-dU, 0.0)]


def plan(model, y_k, y_km1, u_km1, d, Vbar, r, q0, r0, mult=1.0, asymmetric=True, u_max=np.inf,
         u_min=TC_MIN, du_max=DU_MAX, Np=NP, n_iter=1, solver=None):
    """One K-MPC step (module docstring, steps 1-4). Vbar: shifted plan (`shift_plan`).
    Returns V (u_k = V[0]), the solver status, whether the fallback was applied, and Ybar/Phi of
    the last linearisation."""
    solver = solver or Solver()
    Vbar = np.asarray(Vbar, float)
    M = blocking_matrix(Np, len(Vbar))
    V_lin, status = Vbar, None
    for _ in range(n_iter):
        Ybar, Xbar = nominal(model, y_k, y_km1, u_km1, M @ V_lin, d)
        Phi = sensitivity(linearize(model, Xbar)) @ M
        qp = build_qp(Phi, Ybar, V_lin, r, u_km1, q0, r0, mult, asymmetric, u_min, u_max, du_max)
        z, status = solver.solve(qp, warm_start_point(V_lin, u_km1, asymmetric))
        if z is None:
            return {"V": clip_plan(Vbar, u_km1, u_min, u_max, du_max), "status": status, "fallback": True,
                    "Ybar": Ybar, "Phi": Phi}
        V_lin = clip_plan(z[:qp.nv], u_km1, u_min, u_max, du_max)   # removes violations below FEAS_TOL
    return {"V": V_lin, "status": status, "fallback": False, "Ybar": Ybar, "Phi": Phi}


class KMPC:
    """K-MPC with its state between steps: the last plan V (nominal for the next step and u_plan for the
    supervisor) and the filtered disturbance estimate d.

    step(k, y, u, r, info) -> (u_k, details), where y holds the real-time history up to k (what the
    filter has passed by sample k; y[k] is the filter's y_rt), u the applied inputs up to k-1, r the
    set-point (scalar, or one value per horizon step), info = {"mult": supervisor multiplier m_k,
    "missing": y_k not measured, "rejected": y_k rejected by the filter}.

    Disturbance estimate (TZ 5.4 with the changes accepted 10.10.2026): the one-step error
    e_k = y_k - f(x_k-1) is passed through a first-order filter,
        d_k = d_k-1 + beta (e_k - d_k-1),   0 < beta <= 1   (beta = 1: no filter),
    because d_k enters every step of the free run and its noise is amplified along the horizon. A
    constant disturbance is still estimated without bias (the filter's fixed point is d = e). On a
    missing sample, and on a sample the filter rejected (then y_k is the model's own prediction and
    e_k is zero by construction), d_k = d_k-1. beta is tuned on S0 together with rho."""

    def __init__(self, model, u_max, rho, beta=1.0, q0=None, Np=NP, Nc=NC, asymmetric=True, n_iter=1,
                 u_min=TC_MIN, du_max=DU_MAX, hold_d_on_reject=True, solver_settings=None):
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must lie in (0, 1], got {beta}")
        self.model, self.u_max, self.u_min, self.du_max = model, float(u_max), float(u_min), float(du_max)
        self.q0 = q0_default() if q0 is None else float(q0)
        self.rho, self.r0, self.beta = float(rho), r0_of(rho), float(beta)
        self.Np, self.Nc, self.asymmetric, self.n_iter = Np, Nc, asymmetric, n_iter
        self.hold_d_on_reject = hold_d_on_reject
        self.solver = Solver(solver_settings)
        self.V, self.d = None, 0.0
        self.n_steps, self.n_fallback, self.statuses = 0, 0, {}

    def planned_inputs(self, u_prev, n):
        """Inputs planned for the n samples after the last decision (u_k+1 .. u_k+n after the step at k):
        the last plan expanded by the blocking matrix and shifted by one step; u_prev held while there
        is no plan yet."""
        if self.V is None:
            return np.full(n, float(u_prev))
        return (blocking_matrix(n + 1, self.Nc) @ self.V)[1:]

    def update_disturbance(self, k, y, u, info):
        x_km1 = np.array([y[k - 1], y[k - 2], u[k - 1], u[k - 2]], float)
        hold = info.get("missing", False) or (self.hold_d_on_reject and info.get("rejected", False))
        if not hold:
            e = disturbance(self.model, y[k], x_km1)
            self.d = self.d + self.beta * (e - self.d)

    def optimise(self, k, y, u, r, mult, Vbar):
        """The plan for this step: the K-MPC QP (subclasses replace it)."""
        return plan(self.model, y[k], y[k - 1], u[k - 1], self.d, Vbar, r, self.q0, self.r0, mult,
                    self.asymmetric, self.u_max, self.u_min, self.du_max, self.Np, self.n_iter, self.solver)

    def step(self, k, y, u, r, info=None):
        info = info or {}
        self.update_disturbance(k, y, u, info)
        Vbar = shift_plan(self.V, u[k - 1], self.Nc)
        out = self.optimise(k, y, u, r, info.get("mult", 1.0), Vbar)
        self.V = out["V"]
        self.n_steps += 1
        self.n_fallback += int(out["fallback"])
        self.statuses[out["status"]] = self.statuses.get(out["status"], 0) + 1
        y_next = self.model.f(np.array([y[k], y[k - 1], self.V[0], u[k - 1]])) + self.d
        return float(self.V[0]), {"d": self.d, "status": out["status"], "fallback": out["fallback"],
                                  "y_pred_next": float(y_next), "y_used": float(y[k])}
