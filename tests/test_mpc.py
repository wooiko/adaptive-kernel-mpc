"""Tests of part 2 (K-MPC), TZ section 10. Every test names the mutation it must catch
(tests/mutate_mpc.py applies them and records the result in outputs/mutations_mpc.txt).

    python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import closed_loop as cl  # noqa: E402
import controllers as ctl  # noqa: E402
import cstr  # noqa: E402
import data  # noqa: E402
import mpc  # noqa: E402
import pipeline as pl  # noqa: E402
import safety  # noqa: E402
from adaptive import AdaptiveIdentifier, Scaler, regression_set, run_online  # noqa: E402

IDENT = Y = U = SC = X0 = T0 = None
G_REL_TOL = 1e-5          # TZ section 9: G against central differences of the free run


def setUpModule():
    """A window model of the reactor: noise-free CSTR response to a random input over the data range."""
    global IDENT, Y, U, SC, X0, T0
    rng = np.random.default_rng(11)
    U = data.aprbs(400, safety.TC_MIN, 303.0, 10, 40, rng)
    Y = cstr.simulate(U, data.TS, cstr.steady_state(U[0]))[:, 0]
    X0, T0, _ = regression_set(Y, U)
    SC = Scaler().fit(X0, T0)
    IDENT = new_identifier()


def new_identifier(horizon=0):
    """A fresh window model of the module's reactor record (the identifier is changed by a closed-loop run)."""
    return AdaptiveIdentifier(3.0, 1e-4, SC, SC.x(X0), SC.t(T0), data.NOISE_STD, window=400, horizon=horizon)


def _start(rng):
    """Random start (k, y_k, y_k-1, u_k-1) from the record and a random plan of Np inputs."""
    k = int(rng.integers(50, len(Y) - 1))
    Ubar = np.clip(U[k - 1] + np.cumsum(rng.uniform(-1, 1, mpc.NP)), safety.TC_MIN, 303.0)
    return Y[k], Y[k - 1], U[k - 1], Ubar


def _free_run(model, y_k, y_km1, u_km1, Ubar):
    """pipeline.free_run (part 1 code) over the same horizon: an independent path to Ybar."""
    u = np.r_[u_km1, Ubar, Ubar[-1]]
    return pl.free_run(model, np.array([y_km1, y_k]), u)[2:]


class _Shifted:
    """Frozen window model plus a constant: f(x) + d."""
    def __init__(self, d): self.m, self.d = pl.Frozen(IDENT.krr, IDENT.sc), d
    def predict(self, X): return self.m.predict(X) + self.d


def g_rel_error(model_pred, model, y_k, y_km1, u_km1, Ubar, h=1e-3):
    """max |G - G_fd| / max |G|, G_fd from central differences (step h, K) of pipeline.free_run."""
    _, Xbar = mpc.nominal(model, y_k, y_km1, u_km1, Ubar)
    G = mpc.sensitivity(mpc.linearize(model, Xbar))
    Gfd = np.empty_like(G)
    for j in range(len(Ubar)):
        e = np.zeros(len(Ubar)); e[j] = h
        Gfd[:, j] = (_free_run(model_pred, y_k, y_km1, u_km1, Ubar + e)
                     - _free_run(model_pred, y_k, y_km1, u_km1, Ubar - e)) / (2 * h)
    return float(np.max(np.abs(G - Gfd)) / np.max(np.abs(G))), G


class WindowGradient(unittest.TestCase):
    """Mutation: lost factor s_t/s_x in AdaptiveIdentifier.gradient."""
    def test_against_central_differences(self):
        X, _, _ = regression_set(Y, U)
        for x in X[::37]:
            h = 1e-6 * np.maximum(np.abs(x), 1.0)
            fd = np.array([(IDENT.predict(x + hj * e) - IDENT.predict(x - hj * e)) / (2 * hj)
                           for hj, e in zip(h, np.eye(4))])
            g = IDENT.gradient(x)
            self.assertLess(np.linalg.norm(g - fd) / np.linalg.norm(g), 1e-6)


class NominalAndSensitivity(unittest.TestCase):
    """Mutations: column of G shifted by one step; sign of b2; regressor with u_k-1 instead of u_k;
    d_k dropped from the recursion."""
    def setUp(self):
        self.model = mpc.window_model(IDENT)
        self.frozen = pl.Frozen(IDENT.krr, IDENT.sc)

    def test_nominal_equals_free_run(self):
        rng = np.random.default_rng(2)
        for _ in range(3):
            s = _start(rng)
            np.testing.assert_allclose(mpc.nominal(self.model, *s)[0], _free_run(self.frozen, *s),
                                       rtol=0, atol=1e-12)

    def test_disturbance_enters_every_step(self):
        s = _start(np.random.default_rng(3))
        np.testing.assert_allclose(mpc.nominal(self.model, *s, d=0.01)[0], _free_run(_Shifted(0.01), *s),
                                   rtol=0, atol=1e-12)

    def test_G_against_central_differences(self):
        rng = np.random.default_rng(4)
        for _ in range(3):
            err, G = g_rel_error(self.frozen, self.model, *_start(rng))
            self.assertLess(err, G_REL_TOL)
            self.assertTrue(np.all(np.triu(G, 1) == 0))      # causality: u_k+j acts from y_k+j+1 on

    def test_first_input_acts_on_the_next_sample(self):
        """Time convention: Ybar[0] is y_k+1; it depends on u_k through b1 at xbar_k = [y_k, y_k-1, u_k, u_k-1]."""
        y_k, y_km1, u_km1, Ubar = _start(np.random.default_rng(5))
        _, Xbar = mpc.nominal(self.model, y_k, y_km1, u_km1, Ubar)
        np.testing.assert_array_equal(Xbar[0], [y_k, y_km1, Ubar[0], u_km1])
        G = mpc.sensitivity(mpc.linearize(self.model, Xbar))
        self.assertEqual(G[0, 0], self.model.g(Xbar[0])[2])

    def test_linear_model_is_exactly_linear(self):
        m = mpc.linear_model([1.5, -0.56, 0.02, 0.01], 0.3)
        rng = np.random.default_rng(6)
        Ubar, dU = rng.normal(size=mpc.NP), rng.normal(size=mpc.NP)
        Y0, X0 = mpc.nominal(m, 1.0, 0.9, 0.2, Ubar, d=0.05)
        Y1, _ = mpc.nominal(m, 1.0, 0.9, 0.2, Ubar + dU, d=0.05)
        np.testing.assert_allclose(Y1 - Y0, mpc.sensitivity(mpc.linearize(m, X0)) @ dU, atol=1e-12)


class BlockingAndShift(unittest.TestCase):
    """Mutations: input held from move Nc instead of Nc-1; plan not shifted."""
    def test_blocking_matrix(self):
        M = mpc.blocking_matrix(6, 3)
        np.testing.assert_array_equal(M @ np.array([1.0, 2.0, 3.0]), [1, 2, 3, 3, 3, 3])

    def test_shift_plan(self):
        np.testing.assert_array_equal(mpc.shift_plan([1.0, 2.0, 3.0], 0.0, 3), [2, 3, 3])
        np.testing.assert_array_equal(mpc.shift_plan(None, 7.0, 3), [7, 7, 7])


def _qp_case(seed, Nc=mpc.NC, Np=mpc.NP):
    """A realistic QP: Phi from the window model, set-point off the free run."""
    rng = np.random.default_rng(seed)
    y_k, y_km1, u_km1, _ = _start(rng)
    Vbar = np.clip(u_km1 + np.cumsum(rng.uniform(-1, 1, Nc)), safety.TC_MIN, 303.0)
    model, M = mpc.window_model(IDENT), mpc.blocking_matrix(Np, Nc)
    Ybar, Xbar = mpc.nominal(model, y_k, y_km1, u_km1, M @ Vbar)
    Phi = mpc.sensitivity(mpc.linearize(model, Xbar)) @ M
    r = Ybar[-1] + rng.uniform(-0.03, 0.03)
    return Phi, Ybar, Vbar, r, u_km1


def cost(V, Phi, Ybar, Vbar, r, u_km1, q0, r0, mult, asymmetric):
    """J(V) written directly from TZ 5.5 / 6.2 (no matrices from build_qp)."""
    yhat = Ybar + Phi @ (V - Vbar)
    dU = np.diff(np.r_[u_km1, V])
    w = np.where(dU > 0, mult * r0, r0) if asymmetric else mult * r0
    return float(q0 * np.sum((yhat - r) ** 2) + np.sum(w * dU ** 2))


def cost_split(z, Phi, Ybar, Vbar, r, q0, r0, mult, Nc=mpc.NC):
    V, p, n = z[:Nc], z[Nc:2 * Nc], z[2 * Nc:]
    yhat = Ybar + Phi @ (V - Vbar)
    return float(q0 * np.sum((yhat - r) ** 2) + mult * r0 * p @ p + r0 * n @ n)


def num_grad_hess(J, z, h=1e-2):
    n = len(z); E = np.eye(n) * h
    g = np.array([(J(z + E[i]) - J(z - E[i])) / (2 * h) for i in range(n)])
    H = np.array([[(J(z + E[i] + E[j]) - J(z + E[i] - E[j]) - J(z - E[i] + E[j]) + J(z - E[i] - E[j])) / (4 * h * h)
                   for j in range(n)] for i in range(n)])
    return g, H


def feasible(qp, z, tol=1e-9):
    Az = qp.A @ z
    return bool(np.all(Az >= qp.l - tol) and np.all(Az <= qp.u + tol))


Q0, R0, MULT = mpc.q0_default(), 0.3, 1.6


class QPMatrices(unittest.TestCase):
    """Mutation: lost term e u_k-1 in q."""
    def test_symmetric_against_numerical_derivatives(self):
        Phi, Ybar, Vbar, r, u = _qp_case(7)
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asymmetric=False)
        J = lambda V: cost(V, Phi, Ybar, Vbar, r, u, Q0, R0, MULT, False)
        z = Vbar + np.random.default_rng(8).normal(size=len(Vbar))
        g, H = num_grad_hess(J, z)
        np.testing.assert_allclose(H, qp.P, rtol=0, atol=1e-6 * np.abs(qp.P).max())
        np.testing.assert_allclose(g, qp.P @ z + qp.q, rtol=0, atol=1e-6 * np.abs(g).max())

    def test_asymmetric_against_numerical_derivatives(self):
        Phi, Ybar, Vbar, r, u = _qp_case(9)
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asymmetric=True)
        J = lambda z: cost_split(z, Phi, Ybar, Vbar, r, Q0, R0, MULT)
        rng = np.random.default_rng(10)
        z = np.r_[Vbar + rng.normal(size=mpc.NC), rng.uniform(0, 1, 2 * mpc.NC)]
        g, H = num_grad_hess(J, z)
        np.testing.assert_allclose(H, qp.P, rtol=0, atol=1e-6 * np.abs(qp.P).max())
        np.testing.assert_allclose(g, qp.P @ z + qp.q, rtol=0, atol=1e-6 * np.abs(g).max())

    def test_split_cost_equals_asymmetric_cost(self):
        Phi, Ybar, Vbar, r, u = _qp_case(11)
        V = Vbar + np.random.default_rng(12).normal(size=mpc.NC)
        dU = np.diff(np.r_[u, V])
        z = np.r_[V, np.maximum(dU, 0), np.maximum(-dU, 0)]
        self.assertAlmostEqual(cost_split(z, Phi, Ybar, Vbar, r, Q0, R0, MULT),
                               cost(V, Phi, Ybar, Vbar, r, u, Q0, R0, MULT, True), places=9)


class QPConstraints(unittest.TestCase):
    """Mutation: first rate row without u_k-1."""
    def check(self, asymmetric):
        Phi, Ybar, Vbar, r, u = _qp_case(13)
        u = 299.0
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asymmetric, u_max=300.5)
        D = mpc.difference_matrix(mpc.NC)

        def point(V):
            if not asymmetric:
                return V
            dU = D @ V - np.r_[u, np.zeros(mpc.NC - 1)]
            return np.r_[V, np.maximum(dU, 0), np.maximum(-dU, 0)]
        ok = u + np.array([1.0, 1.5, 1.5, 0.5, -0.5])                    # first move exactly +du_max
        self.assertTrue(feasible(qp, point(ok)))
        for bad in (u + np.array([1.01, 1.5, 1.5, 1.5, 1.5]),            # first move above du_max
                    u - np.array([1.01, 1.5, 1.5, 1.5, 1.5]),            # first move below -du_max
                    u + np.array([1.0, 1.6, 1.6, 1.6, 1.6])):            # above u_max = 300.5
            self.assertFalse(feasible(qp, point(bad)))
        u = 292.3                                                          # lower bound alone
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asymmetric, u_max=300.5)
        self.assertTrue(feasible(qp, point(np.full(mpc.NC, 292.0))))
        self.assertFalse(feasible(qp, point(np.full(mpc.NC, 291.9))))

    def test_symmetric(self): self.check(False)
    def test_asymmetric(self): self.check(True)


def scipy_solution(Phi, Ybar, Vbar, r, u, q0, r0, mult, asymmetric, u_min, u_max, du_max):
    """Reference: SLSQP on J(V) from its definition, in deviations from u_k-1."""
    Nc = len(Vbar)
    J = lambda w: cost(w + u, Phi, Ybar, Vbar, r, u, q0, r0, mult, asymmetric)
    rate = lambda w: np.r_[du_max - np.diff(np.r_[0.0, w]), du_max + np.diff(np.r_[0.0, w])]
    res = minimize(J, np.zeros(Nc), method="SLSQP", bounds=[(u_min - u, u_max - u)] * Nc,
                   constraints=[{"type": "ineq", "fun": rate}], options={"ftol": 1e-16, "maxiter": 1000})
    return res.x + u, res.fun


class QPSolution(unittest.TestCase):
    """Mutation: sign of q in the solver."""
    def test_against_scipy_on_random_problems(self):
        solver, active = mpc.Solver(), 0
        for seed in range(20, 30):
            Phi, Ybar, Vbar, r, u = _qp_case(seed)
            asym = seed % 2 == 0
            u_max = u + 1.5
            qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asym, u_max=u_max)
            z, status = solver.solve(qp, mpc.warm_start_point(Vbar, u, asym))
            self.assertEqual(status, "solved")
            Vs, Js = scipy_solution(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asym, safety.TC_MIN, u_max, mpc.DU_MAX)
            Jo = cost(z[:mpc.NC], Phi, Ybar, Vbar, r, u, Q0, R0, MULT, asym)
            np.testing.assert_allclose(z[:mpc.NC], Vs, rtol=0, atol=1e-4)
            self.assertLessEqual(Jo, Js + 1e-9 * max(1.0, abs(Js)))
            Az = qp.A @ z
            active += int(np.any(np.isclose(Az, qp.l, atol=1e-7) | np.isclose(Az, qp.u, atol=1e-7)))
        self.assertGreaterEqual(active, 5)      # the comparison includes active constraints

    def test_unconstrained_equals_closed_form(self):
        Phi, Ybar, Vbar, r, u = _qp_case(31)
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, False, u_min=-np.inf, u_max=np.inf, du_max=np.inf)
        z, status = mpc.Solver().solve(qp)
        self.assertEqual(status, "solved")
        np.testing.assert_allclose(z, -np.linalg.solve(qp.P, qp.q), rtol=0, atol=1e-7)


class AsymmetricPenalty(unittest.TestCase):
    """Mutation: the multiplier also on cooling moves."""
    def solve(self, Phi, Ybar, Vbar, r, u, mult, asym):
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, mult, asym, u_min=-np.inf, u_max=np.inf, du_max=np.inf)
        z, status = mpc.Solver().solve(qp)
        self.assertEqual(status, "solved")
        return z

    def case(self, Nc):
        """Model at its own steady state at u = 297 K: the free run is flat."""
        model, u = mpc.window_model(IDENT), 297.0
        y = Y.mean()
        for _ in range(500):
            y = model.f(np.array([y, y, u, u]))
        Vbar, M = np.full(Nc, u), mpc.blocking_matrix(mpc.NP, Nc)
        Ybar, Xbar = mpc.nominal(model, y, y, u, M @ Vbar)
        return mpc.sensitivity(mpc.linearize(model, Xbar)) @ M, Ybar, Vbar, u

    def test_multiplier_acts_on_heating_only(self):
        """One move (Nc = 1), so its sign is that of the needed change: higher Ca needs cooling
        (dCa/dTc < 0). Cooling: asymmetric with m = 2 equals symmetric with m = 1; heating: m = 2."""
        Phi, Ybar, Vbar, u = self.case(1)
        self.assertTrue(np.all(Phi < 0))
        for dr, same_as, sign in ((+0.02, 1.0, -1.0), (-0.02, 2.0, +1.0)):
            r = Ybar[-1] + dr
            z = self.solve(Phi, Ybar, Vbar, r, u, 2.0, True)
            V_sym = self.solve(Phi, Ybar, Vbar, r, u, same_as, False)
            self.assertGreater(sign * (V_sym[0] - u), 0.0)
            self.assertAlmostEqual(z[0], V_sym[0], places=7)
            V_other = self.solve(Phi, Ybar, Vbar, r, u, 3.0 - same_as, False)
            self.assertGreater(abs(V_other[0] - V_sym[0]), 1e-3)     # the multiplier matters here

    def test_moves_split_into_one_direction(self):
        Phi, Ybar, Vbar, u = self.case(mpc.NC)
        for dr in (0.02, -0.02):
            z = self.solve(Phi, Ybar, Vbar, Ybar[-1] + dr, u, 2.0, True)
            p, n = z[mpc.NC:2 * mpc.NC], z[2 * mpc.NC:]
            self.assertLess(np.max(np.abs(p * n)), 1e-10)
            np.testing.assert_allclose(p - n, np.diff(np.r_[u, z[:mpc.NC]]), atol=1e-9)


class Fallback(unittest.TestCase):
    """Mutation: fallback plan not clipped."""
    def test_solver_failure(self):
        y_k, y_km1, u_km1, _ = _start(np.random.default_rng(51))
        u_km1, u_max = 300.0, 300.5
        Vbar = np.array([300.9, 301.5, 302.0, 302.0, 302.0])         # outside the box and the rate limit
        out = mpc.plan(mpc.window_model(IDENT), y_k, y_km1, u_km1, 0.0, Vbar, Y.mean(), Q0, R0,
                       u_max=u_max, solver=mpc.Solver({**mpc.OSQP_SETTINGS, "max_iter": 1, "polishing": False}))
        self.assertTrue(out["fallback"])
        self.assertNotEqual(out["status"], "solved")
        V = out["V"]
        self.assertTrue(np.all(V <= u_max) and np.all(V >= safety.TC_MIN))
        self.assertTrue(np.all(np.abs(np.diff(np.r_[u_km1, V])) <= mpc.DU_MAX + 1e-12))

    def test_solver_rejects_unsolved_result(self):
        """Without constraints every iterate is feasible: only the status can reject it."""
        Phi, Ybar, Vbar, r, u = _qp_case(52)
        qp = mpc.build_qp(Phi, Ybar, Vbar, r, u, Q0, R0, MULT, False, u_min=-np.inf, u_max=np.inf, du_max=np.inf)
        z, status = mpc.Solver({**mpc.OSQP_SETTINGS, "max_iter": 1, "polishing": False}).solve(qp)
        self.assertIsNone(z)
        self.assertEqual(status, "maximum iterations reached")

    def test_clip_plan_box_wins(self):
        V = mpc.clip_plan([305.0, 305.0], 303.0, u_max=300.5)
        np.testing.assert_array_equal(V, [300.5, 300.5])


class OffsetFree(unittest.TestCase):
    """Mutation: d_k dropped (TZ 5.4, variant A). Linear plant with a constant output disturbance,
    exact model, constraints far away: the steady-state error is zero."""
    def test_output_disturbance(self):
        th = np.array([1.5, -0.56, 0.02, 0.01])            # poles 0.7 and 0.8
        model, solver = mpc.linear_model(th), mpc.Solver()
        N, r, d0 = 300, 0.5, 0.05
        yt, u, V = np.zeros(N + 1), np.zeros(N), None
        for k in range(2, N):
            ym = yt + d0 * (np.arange(N + 1) >= 100)
            d = mpc.disturbance(model, ym[k], [ym[k - 1], ym[k - 2], u[k - 1], u[k - 2]])
            out = mpc.plan(model, ym[k], ym[k - 1], u[k - 1], d, mpc.shift_plan(V, u[k - 1]), r, Q0, 0.1,
                           u_min=-1e3, u_max=1e3, du_max=1e3, solver=solver)
            V = out["V"]; u[k] = V[0]
            yt[k + 1] = th @ [yt[k], yt[k - 1], u[k], u[k - 1]]
        ym = yt + d0
        self.assertLess(np.max(np.abs(ym[N - 50:N] - r)), 1e-6)


def own_saddle_node(p, guess):
    """Saddle-node point from balances written out here (independent of cstr.py): steady state
    f1 = f2 = 0 and det(df/d(Ca, T)) = 0; unknowns Ca, T, Tc."""
    a, J, b = p.q / p.V, p.mdelH / (p.rho * p.Cp), p.UA / (p.V * p.rho * p.Cp)

    def eqs(v):
        Ca, T, Tc = v
        k = p.k0 * np.exp(-p.EoverR / T)
        dk = k * p.EoverR / T ** 2
        f1 = a * (p.Caf - Ca) - k * Ca
        f2 = a * (p.Tf - T) + J * k * Ca + b * (Tc - T)
        det = (-a - k) * (-a + J * dk * Ca - b) - (-dk * Ca) * (J * k)
        return [f1, f2, det]
    return fsolve(eqs, guess, xtol=1e-13)


class TcMax(unittest.TestCase):
    """Mutations: maximum instead of minimum over the vertices; margin with the sign +."""
    @classmethod
    def setUpClass(cls):
        cls.res = safety.tc_max()
        rng = safety.admissible_set()
        grid = [{"UA": ua, "Tf": tf} for ua in np.linspace(*rng["UA"], 4) for tf in np.linspace(*rng["Tf"], 4)]
        cls.grid = np.array([safety.limit_at(g) for g in grid])

    def test_minimum_over_vertices_equals_grid_minimum(self):
        self.assertAlmostEqual(self.res["tc_max"], self.grid.min() - safety.MARGIN, places=9)
        self.assertGreaterEqual(self.grid.min(), self.res["worst_limit"] - 1e-9)

    def test_random_interior_points_not_lower(self):
        rng, s = np.random.default_rng(61), safety.admissible_set()
        for _ in range(8):
            pt = {n: rng.uniform(*s[n]) for n in s}
            self.assertGreater(safety.limit_at(pt), self.res["worst_limit"])

    def test_independent_fsolve(self):
        p = safety.params_at(self.res["worst_vertex"])
        Ca, T, Tc = own_saddle_node(p, [self.res["ca_at_fold"] + 0.01, self.res["T_at_fold"] + 1.0,
                                        self.res["worst_limit"] + 0.5])
        self.assertAlmostEqual(Tc, self.res["worst_limit"], places=6)
        self.assertAlmostEqual(T, self.res["T_at_fold"], places=4)

    def test_no_fold_is_refused(self):
        p0 = cstr.CSTRParams()
        with self.assertRaises(ValueError):
            safety.tc_max({"q": (0.7 * p0.q, p0.q)})

    def test_reachable_range_on_cold_branch(self):
        lo, hi = safety.reachable_setpoints(u_max=self.res["tc_max"])
        tc = np.linspace(safety.TC_MIN, self.res["tc_max"], 12)
        ca = np.array([safety.cold_steady_state(t)[0] for t in tc])
        self.assertTrue(np.all(np.diff(ca) < 0))
        self.assertAlmostEqual(lo, ca[-1], places=12)
        self.assertAlmostEqual(hi, ca[0], places=12)


class PassThrough:
    """Identifier stand-in for the linear-plant tests: every measurement accepted as it is."""
    horizon = 0

    def step(self, x, y, u_plan=None):
        return {"y_keep": y, "flagged": False, "restore": [], "mult": 1.0, "regime": 0, "var_h": np.nan}


TH = np.array([1.5, -0.56, 0.02, 0.01])          # linear test plant, poles 0.7 and 0.8


class _Kick:
    """Wraps a controller and adds `dv` to its decision at sample `at`."""
    def __init__(self, ctrl, at, dv): self.c, self.at, self.dv = ctrl, at, dv
    def planned_inputs(self, u_prev, n): return self.c.planned_inputs(u_prev, n)

    def step(self, k, y, u, r, info):
        v, det = self.c.step(k, y, u, r, info)
        return v + (self.dv if k == self.at else 0.0), det


def _linear_loop(n=60, kick=None):
    ctrl = mpc.KMPC(mpc.linear_model(TH), 1e3, 0.1, u_min=-1e3, du_max=1e3)
    ctrl = _Kick(ctrl, *kick) if kick else ctrl
    r = np.r_[np.zeros(10), np.full(n - 10, 0.5)]
    return cl.run_loop(cl.LinearPlant(TH), np.zeros(3), PassThrough(), ctrl, r, np.zeros(n), np.zeros(n, bool), 0.0)


class ClosedLoopTimeConvention(unittest.TestCase):
    """Mutations: plant driven by u_k-1 instead of u_k; regressor of d_k with u_k-2 instead of u_k-1."""
    def test_exact_model_predicts_the_next_sample(self):
        log = _linear_loop()
        k = np.arange(cl.LAG, len(log["r"]) - 1)
        np.testing.assert_allclose(log["y_pred_next"][k], log["Ca"][k + 1], rtol=0, atol=1e-12)
        self.assertLess(np.nanmax(np.abs(log["d"])), 1e-12)

    def test_input_acts_on_the_next_sample_only(self):
        K, base = 30, _linear_loop()
        kicked = _linear_loop(kick=(K, 1.0))
        self.assertEqual(base["Ca"][K], kicked["Ca"][K])
        self.assertAlmostEqual(kicked["Ca"][K + 1] - base["Ca"][K + 1], TH[2], places=12)


def _cstr_loop(n, add=None, missing=None, hold=False):
    """Short CSTR run with the module's window model and a fresh identifier."""
    ident = new_identifier(horizon=mpc.NP)
    ctrl = mpc.KMPC(mpc.window_model(ident), safety.tc_max()["tc_max"], 1.0, hold_d_on_reject=hold)
    x0 = np.array(safety.cold_steady_state(296.0))
    r = np.r_[np.full(20, x0[0]), np.full(n - 20, x0[0] - 0.02)]
    add = np.random.default_rng(71).normal(0, data.NOISE_STD, n) if add is None else add
    missing = np.zeros(n, bool) if missing is None else missing
    return cl.run_loop(cl.CSTRPlant(), x0, ident, ctrl, r, add, missing, 296.0), add, missing


class ClosedLoopMeasurement(unittest.TestCase):
    """Mutations: controller fed the raw measurement instead of y_rt; restored run not put back into the
    history; d_k recomputed on a missing sample."""
    @classmethod
    def setUpClass(cls):
        n = 80
        add = np.random.default_rng(71).normal(0, data.NOISE_STD, n)
        cls.K = 40
        add[cls.K] += 0.05                                   # a spike of 25 noise std
        missing = np.zeros(n, bool); missing[55:58] = True   # a short gap
        add[65:] += 0.03                                     # a level step: rejected, then restored
        cls.add, cls.missing = add, missing
        cls.log, _, _ = _cstr_loop(n, add, missing)

    def test_controller_gets_the_filtered_value(self):
        log, K = self.log, self.K
        self.assertTrue(log["flagged_rt"][K])
        self.assertEqual(log["y_ctrl"][K], log["y_rt"][K])
        self.assertNotAlmostEqual(log["y_ctrl"][K], log["y_meas"][K], places=3)
        np.testing.assert_array_equal(log["y_ctrl"][cl.LAG:], log["y_rt"][cl.LAG:])

    def test_future_measurements_do_not_change_past_inputs(self):
        """P1: measurements after sample k replaced by NaN leave u[0..k] unchanged."""
        k = 50
        miss = self.missing.copy(); miss[k + 1:] = True
        log2, _, _ = _cstr_loop(len(self.add), self.add, miss)
        np.testing.assert_array_equal(log2["u"][:k + 1], self.log["u"][:k + 1])

    def test_missing_sample_keeps_d(self):
        d = self.log["d"]
        self.assertEqual(d[55], d[54]); self.assertEqual(d[57], d[54])
        self.assertNotEqual(d[54], d[53])

    def test_filter_decisions_equal_open_loop_replay(self):
        """The identifier inside the loop takes the same decisions as adaptive.run_online on the recorded
        measurements and inputs (the supervisor plan differs; the filter does not use it)."""
        log = self.log
        self.assertTrue(log["restored"].any())
        o = run_online(new_identifier(), log["y_meas"], log["u"], log["y_rt"][:cl.LAG])
        np.testing.assert_array_equal(o["flagged_rt"][cl.LAG:], log["flagged_rt"][cl.LAG:])
        np.testing.assert_allclose(o["y_rt"][cl.LAG:], log["y_rt"][cl.LAG:], rtol=0, atol=1e-12)


class KMPCDisturbance(unittest.TestCase):
    """Mutation: d_k recomputed on a missing sample."""
    def test_hold_rules(self):
        m = mpc.linear_model(TH)
        y, u = np.array([0.1, 0.2, 0.4]), np.array([1.0, 1.0, np.nan])
        for hold_rej, info, keeps in ((False, {"missing": True}, True), (False, {"rejected": True}, False),
                                      (True, {"rejected": True}, True), (False, {}, False)):
            c = mpc.KMPC(m, 1e3, 0.1, u_min=-1e3, du_max=1e3, hold_d_on_reject=hold_rej)
            c.d = 0.123
            c.step(2, y, u, 0.0, info)
            if keeps:
                self.assertEqual(c.d, 0.123)
            else:
                self.assertAlmostEqual(c.d, y[2] - TH @ [y[1], y[0], u[1], u[0]], places=12)


class Metrics(unittest.TestCase):
    """Mutation: settling search window starting before the event."""
    def test_known_signals(self):
        n, s = 200, 100
        r = np.r_[np.zeros(s), np.ones(n - s)]
        ca = r.copy()
        ca[s:s + 30] = np.linspace(0.0, 0.99, 30)           # in the band (+-0.005) from sample s + 30 on
        ca[s - 20:s] = 0.5                                  # out of the band before the event only
        ca[s + 40] = 1.004                                  # inside the band
        seg = cl.segment_metrics(ca, r, [s], n)[0]
        self.assertEqual(seg["settling_min"], 30 * data.TS)
        self.assertAlmostEqual(seg["overshoot"], 0.004, places=12)
        self.assertAlmostEqual(seg["ss_error"], 0.0, places=12)
        self.assertAlmostEqual(seg["iae"], np.sum(np.abs(ca[s:] - r[s:])) * data.TS, places=12)
        ca2 = r + 0.01
        seg2 = cl.segment_metrics(ca2, r, [s], n)[0]
        self.assertAlmostEqual(seg2["iae"], 0.01 * (n - s) * data.TS, places=12)
        self.assertIsNone(seg2["settling_min"])
        self.assertAlmostEqual(seg2["ss_error"], 0.01, places=12)

    def test_ignition_flag(self):
        log = {"Ca": np.zeros(5), "r": np.zeros(5), "u": np.full(5, 295.0), "y_meas": np.zeros(5),
               "fallback": np.zeros(5, bool), "status": ["solved"] * 5, "t_ctrl_ms": np.ones(5),
               "t_ident_ms": np.ones(5), "regime": np.zeros(5, int), "flagged_rt": np.zeros(5, bool)}
        for tmax, ign in ((349.9, False), (350.1, True)):
            log["T"] = np.r_[330.0, 331.0, tmax, 332.0, 333.0]
            self.assertEqual(cl.run_metrics(log, [2], 292.0, 300.0, 1.0)["ignition"], ign)


class IgnitionThreshold(unittest.TestCase):
    """Source of IGNITION_T: cold branch below it at Tc_max on every vertex; hot state far above it."""
    def test_threshold_separates_the_branches(self):
        res = safety.tc_max()
        for v in safety.vertices(res["ranges"]):
            self.assertLess(safety.cold_steady_state(res["tc_max"], safety.params_at(v))[1], cl.IGNITION_T)
        hot = cstr.steady_states(cstr.saddle_node()[0] + 0.1)
        self.assertEqual(len(hot), 1)
        self.assertGreater(hot[0][1], cl.IGNITION_T + 10.0)


class DisturbanceFilter(unittest.TestCase):
    """Mutation: filter of d_k bypassed (d_k = e_k)."""
    def test_first_order_recursion(self):
        m = mpc.linear_model(TH)
        c = mpc.KMPC(m, 1e3, 0.1, beta=0.25, u_min=-1e3, du_max=1e3)
        y, u = np.array([0.1, 0.2, 0.0]), np.array([1.0, 1.0, np.nan])
        y[2] = TH @ [y[1], y[0], u[1], u[0]] + 0.04          # one-step error e = 0.04
        expected = 0.0
        for _ in range(5):
            c.update_disturbance(2, y, u, {})
            expected += 0.25 * (0.04 - expected)
            self.assertAlmostEqual(c.d, expected, places=14)

    def test_beta_out_of_range(self):
        for b in (0.0, 1.5):
            with self.assertRaises(ValueError):
                mpc.KMPC(mpc.linear_model(TH), 1e3, 0.1, beta=b)

    def test_rejected_sample_keeps_d_by_default(self):
        c = mpc.KMPC(mpc.linear_model(TH), 1e3, 0.1, u_min=-1e3, du_max=1e3)
        c.d = 0.05
        c.update_disturbance(2, np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0, np.nan]), {"rejected": True})
        self.assertEqual(c.d, 0.05)

    def test_offset_free_with_filter(self):
        """Constant output offset on the measurement of an exactly known linear plant, beta < 1: the
        measured output settles on the set-point."""
        n, d0 = 300, 0.05
        add = np.r_[np.zeros(100), np.full(n - 100, d0)]
        ctrl = mpc.KMPC(mpc.linear_model(TH), 1e3, 0.1, beta=0.2, u_min=-1e3, du_max=1e3)
        log = cl.run_loop(cl.LinearPlant(TH), np.zeros(3), PassThrough(), ctrl, np.full(n, 0.5), add,
                          np.zeros(n, bool), 0.0)
        self.assertLess(np.max(np.abs(log["y_meas"][n - 50:] - 0.5)), 1e-6)


class _Recorder:
    """Controller that holds the input and records the multiplier it receives."""
    def __init__(self): self.mults = []
    def planned_inputs(self, u_prev, n): return np.full(n, u_prev)

    def step(self, k, y, u, r, info):
        self.mults.append(info["mult"])
        return float(u[k - 1]), {}


class _Mult2(PassThrough):
    def step(self, x, y, u_plan=None):
        return {**super().step(x, y, u_plan), "mult": 2.0, "regime": 2}


class SupervisorSwitch(unittest.TestCase):
    """Mutation: supervisor=False ignored."""
    def test_multiplier_passed_or_not(self):
        for sup, want in ((True, 2.0), (False, 1.0)):
            rec = _Recorder()
            cl.run_loop(cl.LinearPlant(TH), np.zeros(3), _Mult2(), rec, np.zeros(10), np.zeros(10),
                        np.zeros(10, bool), 0.0, supervisor=sup)
            self.assertEqual(set(rec.mults), {want})


class PIDController(unittest.TestCase):
    """Mutation: back-calculation term of the anti-windup removed."""
    def test_fopdt_fit_recovers_known_process(self):
        rng = np.random.default_rng(81)
        u = data.aprbs(1500, 292.0, 303.0, 10, 60, rng)
        K, tau, n = -0.02, 1.0, 3
        a = np.exp(-data.TS / tau)
        x = np.zeros(len(u))
        for k in range(len(u) - 1):
            x[k + 1] = a * x[k] + (1 - a) * K * (u[max(k - n, 0)] - u.mean())
        f = ctl.fit_fopdt(x + 0.9, u)
        self.assertEqual(f["delay_samples"], n)
        self.assertAlmostEqual(f["K"], K, delta=1e-3 * abs(K))
        self.assertAlmostEqual(f["tau"], tau, delta=1e-3)
        self.assertAlmostEqual(f["theta"], n * data.TS + data.TS / 2, places=12)

    def test_simc_rules(self):
        Kc, ti = ctl.simc_pi(-0.01, 1.2, 0.45, 0.9)
        self.assertAlmostEqual(Kc, 1.2 / (-0.01 * 1.35), places=12)
        self.assertAlmostEqual(ti, 1.2, places=12)
        self.assertAlmostEqual(ctl.simc_pi(-0.01, 9.0, 0.45, 0.9)[1], 4 * 1.35, places=12)

    def test_anti_windup(self):
        """Linear plant (gain 0.5), set-point out of reach for 30 min, then back: with back-calculation the
        input leaves the limit within a few samples."""
        n, u_max = 400, 0.5
        r = np.r_[np.full(50, 0.1), np.full(300, 1.0), np.full(n - 350, 0.1)]   # 1.0 needs u = 2 > u_max
        Kc, ti = ctl.simc_pi(0.5, 1.0, 0.15, 0.3)
        pid = ctl.PID(Kc, ti, u_max, u_min=-u_max, du_max=1e3)
        log = cl.run_loop(cl.LinearPlant(TH), np.zeros(3), PassThrough(), pid, r, np.zeros(n), np.zeros(n, bool), 0.0)
        u = log["u"]
        self.assertTrue(np.all(np.abs(u[300:350] - u_max) < 1e-12))           # saturated while out of reach
        leave = np.where(u[350:] < u_max - 1e-9)[0][0]
        self.assertLessEqual(leave, 3)


class NMPCOnKRR(unittest.TestCase):
    """Mutation: sign of the move term in the NMPC gradient."""
    def test_gradient_against_central_differences(self):
        Phi, Ybar, Vbar, r, u = _qp_case(91)
        y_k, y_km1, _, _ = _start(np.random.default_rng(91))
        model = mpc.window_model(IDENT)
        V = Vbar + np.random.default_rng(92).normal(0, 0.5, mpc.NC)
        J, g = ctl.nonlinear_cost(model, y_k, y_km1, u, 0.003, V, r, Q0, R0, MULT, True)
        h = 1e-4
        fd = np.array([(ctl.nonlinear_cost(model, y_k, y_km1, u, 0.003, V + h * e, r, Q0, R0, MULT, True)[0]
                        - ctl.nonlinear_cost(model, y_k, y_km1, u, 0.003, V - h * e, r, Q0, R0, MULT, True)[0])
                       / (2 * h) for e in np.eye(mpc.NC)])
        self.assertLess(np.max(np.abs(g - fd)) / np.max(np.abs(g)), 1e-5)

    def test_not_worse_than_kmpc_on_the_true_cost(self):
        model = mpc.window_model(IDENT)
        rng = np.random.default_rng(93)
        for _ in range(3):
            y_k, y_km1, u_km1, _ = _start(rng)
            r = y_k + rng.uniform(-0.03, 0.03)
            k_c = mpc.KMPC(model, 300.5, 1.0); n_c = ctl.NMPC(model, 300.5, 1.0)
            y, u = np.array([y_km1, y_km1, y_k]), np.array([u_km1, u_km1, np.nan])
            vk = k_c.optimise(2, y, u, r, 1.0, mpc.shift_plan(None, u_km1))["V"]
            vn = n_c.optimise(2, y, u, r, 1.0, mpc.shift_plan(None, u_km1))["V"]
            jk = ctl.nonlinear_cost(model, y_k, y_km1, u_km1, 0.0, vk, r, mpc.q0_default(), 1.0, 1.0, True)[0]
            jn = ctl.nonlinear_cost(model, y_k, y_km1, u_km1, 0.0, vn, r, mpc.q0_default(), 1.0, 1.0, True)[0]
            self.assertLessEqual(jn, jk * (1 + 1e-9))


class OracleModel(unittest.TestCase):
    def test_rk4_against_plant_integrator(self):
        x0 = np.array(safety.cold_steady_state(296.0))
        U = np.array([[296.0, 298.0, 299.0, 297.0, 300.0] + [300.0] * 15])
        ca = ctl.rk4_ca(x0, U, cstr.CSTRParams())[0]
        ref = cstr.simulate(np.r_[U[0], U[0, -1]], data.TS, x0)[1:, 0]
        self.assertLess(np.max(np.abs(ca - ref)), 1e-7)


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim = safety.tc_max()["tc_max"]

    def test_streams_distinct_and_new(self):
        ids = list(cl.SCENARIO_IDS.values())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(min(ids), 10)

    def test_s2_setpoint_reachable_under_every_disturbance(self):
        sc = cl.scenario_s2(0, self.lim)
        for k in (0, 150, 300, 500, 999):
            p = sc["plant"].params(k)
            lo, hi = safety.reachable_setpoints(u_max=self.lim, p=p)
            self.assertTrue(lo < sc["r"][k] < hi, k)

    def test_s3_levels_reachable_throughout(self):
        sc = cl.scenario_s3(0, self.lim)
        for k in (sc["starts"][0], sc["starts"][-1], len(sc["r"]) - 1):
            lo, hi = safety.reachable_setpoints(u_max=self.lim, p=sc["plant"].params(k))
            self.assertTrue(np.all((sc["r"][sc["starts"][0]:] > lo) & (sc["r"][sc["starts"][0]:] < hi)))

    def test_s4_reference_differs_only_by_spikes(self):
        a, b = cl.scenario_s4(0, self.lim), cl.scenario_s4(0, self.lim, spikes=False)
        np.testing.assert_allclose(a["add"][~a["spike"]], b["add"][~a["spike"]], atol=1e-15)
        self.assertFalse(b["spike"].any())
        self.assertTrue(np.all(np.abs(a["add"][a["spike"]] - b["add"][a["spike"]]) >= 5 * data.NOISE_STD - 1e-12))

    def test_s5_edge_and_s6_outside(self):
        res = safety.tc_max()
        sc = cl.scenario_s5(0, self.lim)
        edge = safety.cold_steady_state(self.lim, safety.params_at(res["worst_vertex"]))[0]
        self.assertAlmostEqual(sc["r"][-1], edge, places=12)
        p_end = sc["plant"].params(len(sc["r"]) - 1)
        self.assertAlmostEqual(p_end.UA, res["worst_vertex"]["UA"], places=9)
        self.assertAlmostEqual(p_end.Tf, res["worst_vertex"]["Tf"], places=9)
        s6 = cl.scenario_s6(0, self.lim)
        lo, hi = safety.reachable_setpoints(u_max=self.lim)
        self.assertGreater(s6["r"][s6["starts"][0]], hi)
        self.assertLess(s6["r"][s6["starts"][2]], lo)


if __name__ == "__main__":
    unittest.main()
