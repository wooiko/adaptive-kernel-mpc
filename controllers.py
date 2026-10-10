"""Controllers compared with K-MPC (TZ section 7). Common interface, as mpc.KMPC:
    step(k, y, u, r, info) -> (u_k, details);   planned_inputs(u_prev, n) -> inputs planned after k.
All of them use the same hard input constraints (u_min <= Tc <= u_max, |dTc| <= du_max per step) and the
same real-time measurement y (the filter's y_rt).

  PID          PI from a first-order-plus-dead-time model fitted on the commissioning data, SIMC rules,
               anti-windup by back-calculation through the input saturation (box and rate);
  linear MPC   mpc.KMPC on the OE model of part 1 (pipeline.LinearOE): the same QP, weights and d_k;
  NMPC on KRR  the same cost J as K-MPC on the same KRR, minimised without linearisation by SLSQP with
               the exact gradient through G, started from the K-MPC solution;
  NMPC (true)  oracle: the CSTR equations with the exact current parameters and the measured-free true
               state; reference seed only (an upper bound of attainable quality, not a realisable design).
"""
import numpy as np
from scipy.optimize import least_squares, minimize

import cstr
import data
import mpc
from safety import TC_MIN


# ---------------------------------------------------------------- PID (SIMC)

def fit_fopdt(y, u, max_delay=5, ts=data.TS):
    """First order plus dead time, fitted by free-run least squares on a record (cleaned start-up data):
        x_k+1 = a x_k + (1 - a) K u_k-n,  y_k = x_k + c,  a = exp(-Ts / tau).
    The integer delay n = 0 .. max_delay is chosen by the residual; the effective dead time is
    theta = n Ts + Ts/2 (the zero-order hold adds half a sample). Returns K (mol/L per K), tau, theta (min)."""
    y, u = np.asarray(y, float), np.asarray(u, float)
    um = u.mean()

    def sim(p, n):
        K, tau, c = p
        a = np.exp(-ts / max(tau, 1e-6))
        x = np.empty(len(y)); x[0] = y[0] - c
        for k in range(len(y) - 1):
            x[k + 1] = a * x[k] + (1 - a) * K * (u[max(k - n, 0)] - um)
        return x + c
    best = None
    for n in range(max_delay + 1):
        p0 = np.r_[np.polyfit(u - um, y, 1)[0], 1.0, y.mean()]
        r = least_squares(lambda p: sim(p, n) - y, p0, bounds=([-np.inf, 1e-3, -np.inf], [np.inf, 100.0, np.inf]))
        cost = float(np.mean(r.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, n, r.x)
    cost, n, (K, tau, c) = best
    return {"K": float(K), "tau": float(tau), "theta": float(n * ts + ts / 2), "delay_samples": int(n),
            "rmse": float(np.sqrt(cost))}


def simc_pi(K, tau, theta, tau_c):
    """SIMC PI settings (Skogestad 2003): Kc = tau / (K (tau_c + theta)), tau_I = min(tau, 4 (tau_c + theta))."""
    return tau / (K * (tau_c + theta)), min(tau, 4.0 * (tau_c + theta))


class PID:
    """Discrete PI with back-calculation anti-windup (tracking time Tt = tau_I):
        v = Kc e + I,  u = sat(v),  I <- I + Kc Ts / tau_I e + Ts / Tt (u - v),   e = r - y_k,
    sat = the same hard constraints as the MPCs (box, then rate relative to u_k-1). Bumpless start:
    the integral is initialised so that v = u_k-1 at the first step."""
    def __init__(self, Kc, tau_i, u_max, u_min=TC_MIN, du_max=mpc.DU_MAX, ts=data.TS):
        self.Kc, self.tau_i, self.u_max, self.u_min, self.du_max, self.ts = Kc, tau_i, u_max, u_min, du_max, ts
        self.I = None
        self.n_steps, self.n_fallback, self.statuses = 0, 0, {}

    def planned_inputs(self, u_prev, n):
        return np.full(n, float(u_prev))

    def step(self, k, y, u, r, info=None):
        e = float(r - y[k])
        if self.I is None:
            self.I = u[k - 1] - self.Kc * e
        v = self.Kc * e + self.I
        uk = min(max(v, u[k - 1] - self.du_max), u[k - 1] + self.du_max)
        uk = min(max(uk, self.u_min), self.u_max)
        self.I += self.Kc * self.ts / self.tau_i * e + self.ts / self.tau_i * (uk - v)
        self.n_steps += 1
        return float(uk), {"status": "pid", "fallback": False, "y_used": float(y[k])}


# ---------------------------------------------------------------- linear MPC

def linear_mpc(oe, u_max, rho, beta, **kw):
    """mpc.KMPC on the OE model of part 1: y_k+1 = p[:4] . x_k + p[4]."""
    return mpc.KMPC(mpc.linear_model(oe.p[:-1], oe.p[-1]), u_max, rho, beta, **kw)


# ---------------------------------------------------------------- NMPC on the same KRR

def nonlinear_cost(model, y_k, y_km1, u_km1, d, V, r, q0, r0, mult, asymmetric, Np=mpc.NP):
    """J(V) of TZ 5.5 / 6.2 with the nonlinear free run (d inside) instead of its linearisation, and its
    exact gradient: dJ/dV = 2 q0 Phi(V)' (Y - r) + D' (2 w dU), Phi(V) = G(V) M."""
    Nc = len(V)
    M = mpc.blocking_matrix(Np, Nc)
    Y, X = mpc.nominal(model, y_k, y_km1, u_km1, M @ V, d)
    Phi = mpc.sensitivity(mpc.linearize(model, X)) @ M
    dU = np.diff(np.r_[u_km1, V])
    w = np.where(dU > 0, mult * r0, r0) if asymmetric else np.full(Nc, mult * r0)
    rr = np.broadcast_to(np.asarray(r, float), (Np,))
    J = q0 * np.sum((Y - rr) ** 2) + np.sum(w * dU ** 2)
    grad = 2.0 * q0 * Phi.T @ (Y - rr) + mpc.difference_matrix(Nc).T @ (2.0 * w * dU)
    return float(J), grad


class NMPC(mpc.KMPC):
    """NMPC on the KRR of the controller (the same model object as the K-MPC it is compared with):
    SLSQP on J(V) with box and rate constraints, in deviations from u_k-1, started from the K-MPC plan.
    If SLSQP does not report success the K-MPC plan is applied and the step is counted."""
    SLSQP = {"ftol": 1e-12, "maxiter": 200}

    def optimise(self, k, y, u, r, mult, Vbar):
        qp_out = super().optimise(k, y, u, r, mult, Vbar)
        u1, Nc = float(u[k - 1]), self.Nc
        args = (self.model, y[k], y[k - 1], u1, self.d)
        fun = lambda w: nonlinear_cost(*args, w + u1, r, self.q0, self.r0, mult, self.asymmetric, self.Np)
        D = mpc.difference_matrix(Nc)
        cons = [{"type": "ineq", "fun": lambda w: self.du_max - D @ w, "jac": lambda w: -D},
                {"type": "ineq", "fun": lambda w: self.du_max + D @ w, "jac": lambda w: D}]
        res = minimize(fun, qp_out["V"] - u1, jac=True, method="SLSQP", constraints=cons,
                       bounds=[(self.u_min - u1, self.u_max - u1)] * Nc, options=self.SLSQP)
        if not res.success:
            return {**qp_out, "status": "slsqp: " + str(res.message), "fallback": True}
        V = mpc.clip_plan(res.x + u1, u1, self.u_min, self.u_max, self.du_max)
        return {**qp_out, "V": V, "status": "solved", "fallback": False}


# ---------------------------------------------------------------- NMPC on the true model (oracle)

def rk4_ca(x0, U, params, ts=data.TS, sub=10):
    """Ca at the end of each step for a batch of input plans U (n_plans x Np), RK4 with `sub` substeps,
    exact CSTR equations (cstr.rhs written for arrays). Returns (n_plans x Np)."""
    p = params
    a, J, b = p.q / p.V, p.mdelH / (p.rho * p.Cp), p.UA / (p.V * p.rho * p.Cp)
    U = np.atleast_2d(U)
    ca = np.full(len(U), float(x0[0])); T = np.full(len(U), float(x0[1]))
    h = ts / sub
    out = np.empty(U.shape)

    def f(ca, T, tc):
        r = p.k0 * np.exp(-p.EoverR / T) * ca
        return a * (p.Caf - ca) - r, a * (p.Tf - T) + J * r + b * (tc - T)
    for i in range(U.shape[1]):
        tc = U[:, i]
        for _ in range(sub):
            k1 = f(ca, T, tc)
            k2 = f(ca + h / 2 * k1[0], T + h / 2 * k1[1], tc)
            k3 = f(ca + h / 2 * k2[0], T + h / 2 * k2[1], tc)
            k4 = f(ca + h * k3[0], T + h * k3[1], tc)
            ca = ca + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            T = T + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        out[:, i] = ca
    return out


class TrueNMPC:
    """Oracle NMPC: the true state and parameters (info["x_true"], info["params"]), the cost J with the
    symmetric move weight r0 (no disturbance estimate and no supervisor: the model is exact), SLSQP with
    forward-difference gradients (one batch of Nc + 1 simulations), started from the shifted previous plan."""
    def __init__(self, u_max, rho, q0=None, Np=mpc.NP, Nc=mpc.NC, u_min=TC_MIN, du_max=mpc.DU_MAX, h=1e-4):
        self.u_max, self.u_min, self.du_max, self.Np, self.Nc, self.h = u_max, u_min, du_max, Np, Nc, h
        self.q0 = mpc.q0_default() if q0 is None else q0
        self.r0 = mpc.r0_of(rho)
        self.V = None
        self.n_steps, self.n_fallback, self.statuses = 0, 0, {}

    planned_inputs = mpc.KMPC.planned_inputs

    def step(self, k, y, u, r, info):
        x, p = info["x_true"], info["params"]
        u1, Nc, M = float(u[k - 1]), self.Nc, mpc.blocking_matrix(self.Np, self.Nc)
        D = mpc.difference_matrix(Nc)

        def fun(w):
            W = np.vstack([w, w + self.h * np.eye(Nc)])
            Ca = rk4_ca(x, (W + u1) @ M.T, p)
            dU = np.diff(np.c_[np.zeros(len(W)), W], axis=1)
            J = self.q0 * np.sum((Ca - r) ** 2, axis=1) + self.r0 * np.sum(dU ** 2, axis=1)
            return float(J[0]), (J[1:] - J[0]) / self.h
        w0 = mpc.shift_plan(self.V, u1, Nc) - u1
        w0 = mpc.clip_plan(w0 + u1, u1, self.u_min, self.u_max, self.du_max) - u1
        cons = [{"type": "ineq", "fun": lambda w: self.du_max - D @ w, "jac": lambda w: -D},
                {"type": "ineq", "fun": lambda w: self.du_max + D @ w, "jac": lambda w: D}]
        res = minimize(fun, w0, jac=True, method="SLSQP", constraints=cons,
                       bounds=[(self.u_min - u1, self.u_max - u1)] * Nc, options={"ftol": 1e-10, "maxiter": 100})
        ok = bool(res.success)
        self.V = mpc.clip_plan((res.x if ok else w0) + u1, u1, self.u_min, self.u_max, self.du_max)
        self.n_steps += 1
        self.n_fallback += int(not ok)
        st = "solved" if ok else "slsqp: " + str(res.message)
        self.statuses[st] = self.statuses.get(st, 0) + 1
        return float(self.V[0]), {"status": st, "fallback": not ok, "y_used": float(y[k])}
