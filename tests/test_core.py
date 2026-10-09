"""Unit tests: stability limit, LOO shortcut, filter state machine, supervisor, log checks.

    python -m unittest discover -s tests
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks  # noqa: E402
import cstr  # noqa: E402
import data  # noqa: E402
import figures  # noqa: E402
from adaptive import (LAG, N_JUMP, AdaptiveIdentifier, Scaler, Supervisor, regression_set,  # noqa: E402
                      regressor, run_online)
from kernel import KRR, loo_grid_search  # noqa: E402


class StabilityLimit(unittest.TestCase):
    CASES = ({}, {"UA": 0.9}, {"k0": 0.8}, {"Caf": 1.05}, {"Caf": 0.95}, {"q": 0.9})

    def test_saddle_node_conditions(self):
        for ch in self.CASES:
            p = replace(cstr.CSTRParams(), **{k: getattr(cstr.CSTRParams(), k) * v for k, v in ch.items()})
            r = checks.saddle_node(p)
            self.assertLess(r["f"], 1e-8, ch)
            self.assertLess(r["df_dT"], 1e-5, ch)

    def test_cold_state_exists_only_below_the_limit(self):
        for ch in self.CASES:
            p = replace(cstr.CSTRParams(), **{k: getattr(cstr.CSTRParams(), k) * v for k, v in ch.items()})
            Tc, _, T = cstr.saddle_node(p)
            below = cstr.steady_states(Tc - 0.05, p, 250.0, 500.0, 50001)
            above = cstr.steady_states(Tc + 0.05, p, 250.0, 500.0, 50001)
            self.assertLess(below[0][1], T, ch)          # a state colder than the fold exists below it
            self.assertGreater(above[0][1], T, ch)       # and none above it

    def test_derivative_of_tc_curve(self):
        T = np.linspace(310.0, 380.0, 15)
        h = 1e-5
        fd = (cstr.tc_of_T(T + h)[0] - cstr.tc_of_T(T - h)[0]) / (2 * h)
        np.testing.assert_allclose(cstr.tc_of_T(T)[2], fd, rtol=1e-6, atol=1e-8)


class LeaveOneOut(unittest.TestCase):
    def test_grid_shortcut_equals_closed_form(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 3)); y = np.sin(X).sum(1) + 0.1 * rng.normal(size=120)
        for s, lam in ((0.7, 1e-3), (2.0, 1e-1)):
            g = loo_grid_search(X, y, [s], [lam])[2][0, 0]
            e = KRR(s, lam).fit(X, y).loo_residuals()
            self.assertAlmostEqual(g, np.sqrt(np.mean(e ** 2)), places=10)


def _linear_process(n, rng, noise=0.01, hold=60):
    """Stable second-order linear process (poles 0.906 and -0.11), input held for `hold` samples."""
    u = np.repeat(rng.uniform(-1, 1, n // hold + 1), hold)[:n]
    y = np.zeros(n)
    for k in range(n - 1):
        y[k + 1] = 0.8 * y[k] + 0.1 * y[k - 1] + 0.3 * u[k]
    return y + rng.normal(0, noise, n), u


def _identifier(rng, noise=0.01, **kw):
    y, u = _linear_process(600, rng, noise)
    X, t, _ = regression_set(y, u)
    sc = Scaler().fit(X, t)
    return AdaptiveIdentifier(2.0, 1e-2, sc, sc.x(X), sc.t(t), noise, window=500, thr=6.0, **kw)


class FilterStateMachine(unittest.TestCase):
    # the input changes at multiples of 60; events are placed 30 samples after a change,
    # when the response has settled (0.906 ** 30 = 0.05), so they are not confused with it
    K = 150

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.y, self.u = _linear_process(240, self.rng)

    def run_(self, y, **kw):
        return run_online(_identifier(np.random.default_rng(3), **kw), y, self.u, y[:LAG])

    def test_single_spike_rejected_and_replaced(self):
        K = self.K
        y = self.y.copy(); y[K] += 0.5
        o = self.run_(y)
        self.assertTrue(o["flagged_rt"][K] and o["flagged"][K])
        self.assertFalse(o["jump"][K])
        self.assertAlmostEqual(o["y_rt"][K], o["pred"][K])
        self.assertAlmostEqual(o["y_clean"][K], o["pred"][K])

    def test_level_step_restored_as_process_change(self):
        K = self.K
        y = self.y.copy(); y[K:] += 0.5
        o = self.run_(y)
        run = np.arange(K, K + N_JUMP - 1)
        self.assertTrue(o["flagged_rt"][run].all())               # rejected at the time
        self.assertFalse(o["flagged"][K:K + N_JUMP].any())         # restored afterwards
        self.assertTrue(o["jump"][K + N_JUMP - 1])
        np.testing.assert_allclose(o["y_rt"][run], o["pred"][run])  # what the model got in real time
        np.testing.assert_allclose(o["y_clean"][K:K + 10], y[K:K + 10])
        # after the change the trend test uses only samples from the new level: nothing rejected
        self.assertFalse(o["flagged_rt"][K + N_JUMP:K + N_JUMP + 10].any())

    def test_spike_in_jump_mode_caught_by_trend_test(self):
        y = self.y.copy(); y[self.K:] += 0.5
        k = self.K + N_JUMP + 6
        y[k] += 0.5
        o = self.run_(y, adapt=False)              # frozen filter: stays in jump mode
        self.assertTrue(o["jump"][k])
        self.assertTrue(o["flagged"][k])
        self.assertNotAlmostEqual(o["y_clean"][k], y[k])

    def test_missing_sample_inside_a_run(self):
        K = self.K
        y = self.y.copy(); y[K:] += 0.5; y[K + 1] = np.nan
        o = self.run_(y, adapt=False)              # frozen model: its predictions can be reproduced
        self.assertFalse(o["flagged_rt"][K + 1])
        self.assertAlmostEqual(o["y_rt"][K + 1], o["pred"][K + 1])
        # after the restore the gap is predicted again, from the restored history
        ref = _identifier(np.random.default_rng(3), adapt=False)
        self.assertAlmostEqual(o["y_clean"][K + 1], ref.predict(regressor(o["y_clean"], self.u, K)))
        self.assertNotAlmostEqual(o["y_clean"][K + 1], o["y_rt"][K + 1])
        self.assertFalse(o["jump"][K + N_JUMP - 1])   # the missing sample does not count towards the run
        self.assertTrue(o["jump"][K + N_JUMP])
        self.assertTrue(np.isfinite(o["y_clean"]).all())


class SupervisorHysteresis(unittest.TestCase):
    def test_regimes(self):
        s = Supervisor(0.1, hold=3)
        reg = [s.update(v)[0] for v in (0.05, 0.2, 0.5, 0.05, 0.05, 0.05, 0.05)]
        self.assertEqual(reg, [0, 1, 2, 2, 2, 0, 0])
        s = Supervisor(0.1)
        self.assertAlmostEqual(s.update(0.2)[1], 1.5)      # cautious: 1 + (var - th1) / (th2 - th1)
        self.assertEqual(s.update(0.4)[1], 2.0)


class LogChecks(unittest.TestCase):
    def log(self, n=50):
        return pd.DataFrame({"t_min": np.arange(n) * 0.1, "Tc_K": np.linspace(292, 303, n),
                             "Ca_mol_L": np.full(n, 0.9)})

    def test_valid(self):
        self.assertAlmostEqual(data.validate_log(self.log(), 10), 0.1)

    def test_rejects(self):
        cases = []
        d = self.log(); d.loc[5, "Tc_K"] = np.nan; cases.append(d)
        d = self.log(); d.loc[10:, "t_min"] += 0.05; cases.append(d)
        cases.append(self.log().drop(columns="Tc_K"))
        d = self.log(); d["Tc_K"] = 300.0; cases.append(d)
        d = self.log(); d["Ca_mol_L"] = np.nan; cases.append(d)
        for d in cases:
            with self.assertRaises(ValueError):
                data.validate_log(d, 10)
        with self.assertRaises(ValueError):
            data.validate_log(self.log(5), 10)


class RollingRMSE(unittest.TestCase):
    def test_edges_and_gaps(self):
        e = np.ones(1000); e[400:420] = np.nan
        r = figures._rolling_rmse(e, 300)
        np.testing.assert_allclose(r, 1.0)


if __name__ == "__main__":
    unittest.main()
