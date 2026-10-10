"""Unit tests: stability limit, LOO shortcut, filter state machine, supervisor, log checks.

    python -m unittest discover -s tests
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks  # noqa: E402
import cstr  # noqa: E402
import data  # noqa: E402
import figures  # noqa: E402
import pipeline as pl  # noqa: E402
from adaptive import (LAG, N_JUMP, AdaptiveIdentifier, Scaler, Supervisor, first_opportunity,  # noqa: E402
                      regression_set, regressor, run_online)
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

    def test_no_limit_without_multiplicity(self):
        p = replace(cstr.CSTRParams(), q=cstr.CSTRParams().q * 0.7)
        self.assertTrue(np.isnan(cstr.saddle_node(p)[0]))

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


    def test_trend_restarts_after_a_change_found_by_the_trend_test(self):
        K, K2 = self.K, self.K + N_JUMP + 12
        y = self.y.copy(); y[K:] += 0.5; y[K2:] += 0.5
        o = self.run_(y, adapt=False)                 # frozen filter: stays in jump mode
        self.assertTrue(o["jump"][K2:K2 + 20].all())
        self.assertTrue(o["flagged_rt"][K2:K2 + N_JUMP - 1].all())     # the trend test rejects the new level ...
        self.assertFalse(o["flagged"][K2:K2 + N_JUMP].any())            # ... and restores it as a change
        self.assertFalse(o["flagged_rt"][K2 + N_JUMP:K2 + N_JUMP + 8].any())   # then follows the new level


class FirstOpportunity(unittest.TestCase):
    def test_plan_first_contains_an_input_change(self):
        H, j = 6, 120
        ident = _identifier(np.random.default_rng(3), adapt=False, horizon=H)
        u = np.zeros(160); u[j:] = 5.0                  # far outside the training inputs (-1 .. 1)
        y = np.zeros(160)
        o = run_online(ident, y, u, y[:LAG])
        k = first_opportunity(j, H)
        self.assertLess(o["var_h"][k - 1], 0.5)
        self.assertGreater(o["var_h"][k], 0.5)


class Tuning(unittest.TestCase):
    def setUp(self):
        self.y, self.u = _linear_process(1500, np.random.default_rng(0))
        X, t, _ = regression_set(self.y, self.u)
        self.sc = Scaler().fit(X, t)
        self.Z, self.tz = self.sc.x(X), self.sc.t(t)

    def test_second_stage_holdout(self):
        n_fit = int((1 - pl.HOLDOUT) * len(self.Z))
        seen = []

        def loo(Zs, ts, sigmas, lams):          # two LOO candidates: the largest and the smallest lam
            tab = np.full((len(sigmas), len(lams)), 10.0)
            tab[0, -1] = tab[0, 0] = 1.0
            return None, None, tab

        def free_run(m, y_ho, u_ho):            # the largest lam diverges on the held-out block
            seen.append((y_ho, u_ho))
            out = y_ho.copy()
            out[LAG:] += 1.0 if m.krr.lam == pl.LAMS[-1] else 0.01
            return out

        with mock.patch.object(pl, "loo_grid_search", loo), mock.patch.object(pl, "free_run", free_run):
            sigma, lam, _, idx, info = pl.tune(self.Z, self.tz, self.y, self.u, self.sc, n_max=300)
        self.assertEqual(lam, pl.LAMS[0])
        self.assertTrue(info["changed_by_free_run"])
        self.assertLess(idx.max(), n_fit)                       # LOO rows never reach the held-out block
        for y_ho, u_ho in seen:                                 # the block starts LAG samples before its first target
            np.testing.assert_array_equal(y_ho, self.y[n_fit:])
            np.testing.assert_array_equal(u_ho, self.u[n_fit:])

    def test_calibration_rows_match_their_targets(self):
        """With the exact one-step predictor in place of the SVR, the out-of-sample residual is the noise
        only; a row matched to the wrong target would add the process response to it."""
        sc = self.sc
        b = np.array([0.8, 0.1, 0.3, 0.0])

        class Oracle:
            def predict(self, Zq):
                return sc.t((Zq * sc.sx + sc.mx) @ b)

        with mock.patch.object(pl, "fit_svr", lambda *a, **k: Oracle()), mock.patch.object(pl, "MIN_CAL_ROWS", 500):
            thr, _, n = pl.calibrate_threshold(self.y, self.u, self.Z, self.tz, sc, 2.0, 0.01, window=200)
        self.assertGreater(n, 1000)
        self.assertLess(thr[0.005], 4.0)          # |N(0,1)| quantile 0.995 = 2.8


    def test_calibration_predicts_only_rows_after_the_svr_window(self):
        log = []

        class Recorder:
            def __init__(self, Z): self.Z = Z
            def predict(self, Zq):
                log.append((self.Z, Zq))
                return np.zeros(len(Zq))

        with mock.patch.object(pl, "fit_svr", lambda Z, *a, **k: Recorder(Z)), mock.patch.object(pl, "MIN_CAL_ROWS", 0):
            pl.calibrate_threshold(self.y, self.u, self.Z, self.tz, self.sc, 2.0, 0.01, window=200)
        out_of_sample = [(Z, Zq) for Z, Zq in log if len(Zq) < len(Z)]
        self.assertGreater(len(out_of_sample), 50)
        for Z, Zq in out_of_sample:
            train = {r.tobytes() for r in Z}
            self.assertFalse(any(r.tobytes() in train for r in Zq))


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
        for d in cases:
            with self.assertRaises(ValueError):
                data.validate_log(d, 10)
        with self.assertRaises(ValueError):
            data.validate_log(self.log(5), 10)


class Resolution(unittest.TestCase):
    def test_coarse_recording_step_is_refused(self):
        rng = np.random.default_rng(5)
        y = np.sin(np.arange(3000) / 50.0) + rng.normal(0, 0.002, 3000)
        noise = data.noise_std_estimate(y)
        self.assertAlmostEqual(noise, 0.002, delta=0.0003)
        data.check_resolution(y, noise)                        # passes: recorded at full precision
        yq = np.round(y / 0.008) * 0.008
        with self.assertRaises(ValueError):
            data.check_resolution(yq, data.noise_std_estimate(yq))

    def test_rounded_time_stamps_pass_and_a_lost_row_fails(self):
        n = 50
        d = pd.DataFrame({"t_min": np.round(np.arange(n) / 60.0, 3), "Tc_K": np.linspace(292, 303, n),
                          "Ca_mol_L": np.full(n, 0.9)})
        self.assertAlmostEqual(data.validate_log(d, 10), 1 / 60.0, places=4)
        with self.assertRaises(ValueError):
            data.validate_log(d.drop(index=20).reset_index(drop=True), 10)


class StartupCheck(unittest.TestCase):
    def test_short_or_empty_startup_block(self):
        y, u = _linear_process(1500, np.random.default_rng(2))
        with self.assertRaisesRegex(ValueError, "needs about"):
            pl.startup_check(y, u, 0.625)                       # 1500 < window + calibration rows
        with self.assertRaises(ValueError):
            pl.startup_check(np.full(3000, np.nan), np.repeat([0.0, 1.0], 1500))


class RollingRMSE(unittest.TestCase):
    def test_edges_and_gaps(self):
        e = np.ones(1000); e[400:420] = np.nan
        r = figures._rolling_rmse(e, 300)
        np.testing.assert_allclose(r, 1.0)


if __name__ == "__main__":
    unittest.main()
