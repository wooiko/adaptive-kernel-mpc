"""Three-level adaptive identification, one sample at a time.

    level 1  SVR filter      rejects anomalous measurements
    level 2  KRR on a sliding window   follows slow drift of the process
    level 3  GP supervisor   measures how far the operating point, and the trajectory the
                             controller plans from it, are from the data the model was
                             trained on, and makes the controller cautious

All three use the same RBF kernel. The model is NARX:
    y[k+1] = f( y[k], y[k-1], u[k], u[k-1] )
"""
import copy
from collections import deque

import numpy as np
from sklearn.svm import SVR

from kernel import KRR

NA, NB = 2, 2            # past outputs / past inputs in the regressor
LAG = max(NA, NB)
N_JUMP = 4               # consecutive rejections taken as a process change
REFIT_EVERY_JUMP = 5     # accepted samples between refits while in jump mode


def first_opportunity(j, horizon):
    """First sample at which the supervisor's planned trajectory contains the input applied
    at sample j: the plan checked at step k holds inputs up to u[k + horizon - 1] and its
    result is recorded at sample k + 1 (see AdaptiveIdentifier, horizon)."""
    return j - horizon + 2


def regressor(y_hist, u_hist, k):
    """Regressor for predicting y[k+1] from data up to time k."""
    return np.r_[y_hist[k - NA + 1:k + 1][::-1], u_hist[k - NB + 1:k + 1][::-1]]


def regression_set(y, u):
    """All (regressor, target) pairs of a record; rows with missing values dropped."""
    ks = np.arange(LAG - 1, len(y) - 1)
    X = np.array([regressor(y, u, k) for k in ks])
    t = y[ks + 1]
    ok = ~np.isnan(X).any(1) & ~np.isnan(t)
    return X[ok], t[ok], ks[ok] + 1


class Scaler:
    """z-score normalisation of regressors and output."""
    def fit(self, X, t):
        self.mx, self.sx, self.mt, self.st = X.mean(0), X.std(0), t.mean(), t.std()
        return self

    def x(self, X): return (X - self.mx) / self.sx
    def t(self, t): return (t - self.mt) / self.st
    def t_inv(self, z): return z * self.st + self.mt


def robust_std(r):
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def fit_svr(Z, tz, sigma, noise_z, C=10.0):
    """epsilon-insensitive regression: errors inside the +-epsilon tube cost nothing and
    large errors cost only linearly, so isolated spikes barely move the fit."""
    return SVR(kernel="rbf", gamma=1.0 / (2.0 * sigma ** 2), C=C, epsilon=noise_z).fit(Z, tz)


class Supervisor:
    """Turns the GP variance into an operating regime and a control-penalty multiplier.

    normal        var < th1           R = R0
    cautious      th1 <= var < th2    R = R0 * (1 + (var - th1) / (th2 - th1)), continuous at th1
    conservative  var >= th2          R = 2 R0
    with th1 = lam (the regularisation level) and th2 = 3 th1. Stepping back to a
    calmer regime requires `hold` consecutive samples below the threshold.
    The thresholds are a heuristic: the prior variance of the GP is fixed at 1, not
    estimated, so the variance is a measure of data coverage, not a calibrated error."""
    NAMES = ("normal", "cautious", "conservative")

    def __init__(self, lam, hold=3):
        self.th1, self.th2, self.hold = lam, 3.0 * lam, hold
        self.regime, self._calm = 0, 0

    def update(self, var):
        raw = 0 if var < self.th1 else (1 if var < self.th2 else 2)
        if raw >= self.regime:
            self.regime, self._calm = raw, 0
        else:
            self._calm += 1
            if self._calm >= self.hold:
                self.regime, self._calm = raw, 0
        ramp = min(max((var - self.th1) / (self.th2 - self.th1), 0.0), 1.0)
        mult = (1.0, 1.0 + ramp, 2.0)[self.regime]
        return self.regime, mult


class AdaptiveIdentifier:
    """Online identifier.

    Anomaly test: |measurement - SVR prediction| > thr * res_std, where res_std is the
    robust in-sample scatter of the current SVR and `thr` is calibrated at start-up on
    out-of-sample residuals (pipeline.calibrate_threshold), so that a chosen share of
    normal samples is rejected. thr is a quantile ratio, not a Gaussian 'number of sigmas'.

    A run of n_jump consecutive rejections is taken as a change of the process ('jump'):
      * the rejected samples of the run are handed back (key 'restore') so that the caller
        puts the measurements back into the history and the window (method `learn`);
      * while in jump mode the SVR test is not trusted, and single spikes are caught by a
        model-free test: distance from the least-squares line through the last `n_trend`
        accepted measurements since the change (the restored run included), extrapolated
        to the current sample, above thr_jump times
        its standard deviation for white sensor noise, noise_std * sqrt(1 + 1/m +
        (t - t_mean)^2 / Stt). A run of n_jump such rejections is again restored as a change;
      * the models are refitted every `refit_every_jump` accepted samples;
      * jump mode ends after n_jump consecutive samples on which the SVR agrees again.
        With adapt=False the SVR is frozen, so jump mode lasts until the process returns
        to where the frozen filter agrees with it; the model-free test guards it meanwhile.

    horizon > 0: the supervisor also receives the largest GP variance along a free-run
    of the model under the planned input (method `step`, argument u_plan): `horizon`
    regressors, from the current one (data up to k) on, with inputs up to u[k+H-1].
    This is the uncertainty of the trajectory a predictive controller would rely on, not
    only of the current point. The result is recorded at sample k+1, so an input change
    at sample j can first be seen at sample j - H + 2, H - 2 samples before it."""

    def __init__(self, sigma, lam, scaler, Z0, tz0, noise_std, window=500, thr=3.5,
                 n_jump=N_JUMP, use_filter=True, adapt=True, refit_every=10, refit_every_jump=REFIT_EVERY_JUMP,
                 thr_jump=4.0, n_trend=6, horizon=0):
        self.sigma, self.lam, self.sc = sigma, lam, scaler
        self.thr, self.n_jump, self.thr_jump = thr, n_jump, thr_jump
        self.use_filter, self.adapt = use_filter, adapt
        self.refit_every, self.refit_every_jump, self.horizon = refit_every, refit_every_jump, horizon
        self.noise_std = noise_std
        self.noise_z = noise_std / scaler.st
        self.Z, self.tz = deque(Z0[-window:], maxlen=window), deque(tz0[-window:], maxlen=window)
        self.krr = KRR(sigma, lam).fit(np.array(self.Z), np.array(self.tz))
        self.supervisor = Supervisor(lam)
        self._consec, self._jump, self._passed, self._since = 0, False, 0, 0
        self._t, self._pending = 0, []          # sample counter; samples rejected in the current run
        self._acc = deque(maxlen=n_trend)     # (sample counter, measurement) of the last accepted samples
        self._mf_pending = []                 # samples rejected by the model-free test in the current run
        self._refit_svr()

    def _refit_svr(self):
        Z, tz = np.array(self.Z), np.array(self.tz)
        self.svr = fit_svr(Z, tz, self.sigma, self.noise_z)
        self.res_std = robust_std(tz - self.svr.predict(Z))   # in-sample scatter, scaled by thr

    def _refit(self):
        self.krr.fit(np.array(self.Z), np.array(self.tz))
        if self.use_filter:
            self._refit_svr()
        self._since = 0

    def learn(self, X, y, back):
        """Add measured samples (physical-unit regressors X, outputs y, taken `back` samples
        ago) to the window."""
        for x, v, b in sorted(zip(X, y, back), key=lambda r: -r[2]):
            self._acc.append((self._t - b, float(v)))
            if self.adapt:
                self.Z.append(self.sc.x(x)); self.tz.append(self.sc.t(v))
                self._since += 1
        if self.adapt and self._since:
            self._refit()

    def predict(self, x):
        """One-step prediction of the current window model, physical units."""
        return float(self.sc.t_inv(self.krr.predict(self.sc.x(np.asarray(x, float))[None, :])[0]))

    def gradient(self, x):
        """d y[k+1] / d x of the current window model at regressor x, physical units (as
        pipeline.StaticKRR.gradient): dy/dx_j = (st / sx_j) * df/dz_j. Used by the MPC."""
        return self.krr.gradient(self.sc.x(np.asarray(x, float))) * self.sc.st / self.sc.sx

    def horizon_variance(self, x, u_plan):
        """Largest GP variance along an H-step free-run from regressor x (data up to k)
        with the planned inputs u_plan = u[k+1 .. k+H-1]. Output and input lags are
        shifted as the model predicts; one Cholesky solve for all H points."""
        H = len(u_plan) + 1
        Xs, xc = np.empty((H, len(x))), np.array(x, float)
        for h in range(H):
            Xs[h] = xc
            if h < H - 1:
                y_next = float(self.sc.t_inv(self.krr.predict(self.sc.x(xc)[None, :])[0]))
                xc = np.r_[y_next, xc[:NA - 1], u_plan[h], xc[NA:NA + NB - 1]]
        return float(np.max(self.krr.variance(self.sc.x(Xs))))

    def _spike_model_free(self, y_meas):
        if len(self._acc) < 3:
            return False, np.nan
        t, v = np.array(self._acc).T
        tm = t.mean(); stt = np.sum((t - tm) ** 2)
        b = np.sum((t - tm) * (v - v.mean())) / stt
        extrap = v.mean() + b * (self._t - tm)
        sd = self.noise_std * np.sqrt(1.0 + 1.0 / len(t) + (self._t - tm) ** 2 / stt)
        return abs(y_meas - extrap) > self.thr_jump * sd, extrap

    def step(self, x, y_meas, u_plan=None):
        """x: regressor built from data up to k; y_meas: measurement of y[k+1] (may be NaN);
        u_plan: planned inputs u[k+1 .. k+H-1] (used only if horizon > 0).
        Returns prediction, anomaly flag, the value to keep in the history, GP variance at
        the current point and along the horizon, regime, penalty multiplier and 'restore':
        how many samples back (0 = this one) lie the measurements that must be put back
        into the history and passed to `learn`."""
        self._t += 1
        z = self.sc.x(x)[None, :]
        pred = float(self.sc.t_inv(self.krr.predict(z)[0]))
        var = float(self.krr.variance(z)[0])
        var_h = self.horizon_variance(x, u_plan) if self.horizon and u_plan is not None else var
        regime, mult = self.supervisor.update(max(var, var_h))
        missing = np.isnan(y_meas)
        flagged, restore, y_keep = False, [], y_meas
        if not missing and self.use_filter:
            delta = abs(self.sc.t(y_meas) - self.svr.predict(z)[0])
            over = delta > self.thr * self.res_std
            if self._jump:
                spike, extrap = self._spike_model_free(y_meas)
                if spike and len(self._mf_pending) + 1 >= self.n_jump:
                    restore = [self._t - t for t in self._mf_pending] + [0]
                    self._mf_pending = []
                    self._acc.clear()        # the trend restarts from the restored samples (method learn)
                elif spike:
                    flagged, y_keep = True, extrap
                    self._mf_pending.append(self._t)
                else:
                    self._mf_pending = []
                    self._passed = 0 if over else self._passed + 1
                    if self._passed >= self.n_jump:
                        self._jump, self._consec = False, 0
            elif over:
                self._consec += 1
                if self._consec >= self.n_jump:
                    self._jump, self._passed = True, 0
                    restore = [self._t - t for t in self._pending] + [0]
                    self._pending = []
                    self._acc.clear()        # samples before the change do not belong to the new trend
                else:
                    flagged = True
                    self._pending.append(self._t)
            else:
                self._consec, self._pending = 0, []
        if missing:
            y_keep = pred
        elif flagged and not self._jump:
            y_keep = pred
        accept = not missing and not flagged
        if accept and not restore:
            self._acc.append((self._t, float(y_meas)))
            if self.adapt:
                self.Z.append(z[0]); self.tz.append(self.sc.t(y_meas))
                self._since += 1
                if self._since >= (self.refit_every_jump if self._jump else self.refit_every):
                    self._refit()
        return {"pred": pred, "flagged": flagged, "jump": self._jump, "restore": restore,
                "y_keep": y_keep, "var": var, "var_h": var_h, "regime": regime, "mult": mult}


def run_online(ident, y_meas, u, y_init, snapshot_at=(), probe_u=()):
    """Feed a record through the identifier. The regressor always uses the cleaned
    history (rejected or missing samples are replaced by the model prediction; when a
    run of rejections turns out to be a process change, the measurements are restored).
    y_init: values for the first LAG samples of the history (normally the first LAG
    measurements of this record).
    Two sets of flags: `flagged_rt` is the decision taken at each sample, i.e. what a
    controller receives; `flagged` is the final one, after runs of rejections recognised
    as process changes were restored (`y_rt` / `y_clean` likewise).
    snapshot_at: sample indices at which a copy of the
    window model is stored, for frozen-model tests. probe_u: input values at which the GP
    variance is also evaluated at every sample (current outputs, both input lags = probe)."""
    n = len(u)
    H = getattr(ident, "horizon", 0)
    y_hist = np.array(y_meas, dtype=float)
    y_hist[:LAG] = y_init[:LAG]
    out = {k: np.full(n, np.nan) for k in ("pred", "var", "var_h", "mult")}
    out["flagged"], out["regime"], out["jump"] = np.zeros(n, bool), np.zeros(n, int), np.zeros(n, bool)
    out["flagged_rt"] = np.zeros(n, bool)     # decision at the moment of the sample (never undone)
    out["y_rt"] = np.full(n, np.nan)          # value put into the history at that moment
    out["y_rt"][:LAG] = y_hist[:LAG]
    out["snapshots"] = {}
    out["probe_var"] = np.full((n, len(probe_u)), np.nan)
    for k in range(LAG - 1, n - 1):
        if k + 1 in snapshot_at:
            out["snapshots"][k + 1] = copy.deepcopy(ident.krr)
        u_plan = u[k + 1:k + H] if H else None
        if H and len(u_plan) < H - 1:                  # end of record: hold the last input
            u_plan = np.r_[u_plan, np.full(H - 1 - len(u_plan), u[-1])]
        x = regressor(y_hist, u, k)
        if len(probe_u):
            P = np.repeat(x[None, :], len(probe_u), 0)
            P[:, NA:] = np.asarray(probe_u)[:, None]
            out["probe_var"][k + 1] = ident.krr.variance(ident.sc.x(P))
        r = ident.step(x, y_meas[k + 1], u_plan)
        y_hist[k + 1] = out["y_rt"][k + 1] = r["y_keep"]
        out["flagged_rt"][k + 1] = r["flagged"]
        for key in ("pred", "var", "var_h", "mult", "flagged", "regime", "jump"):
            out[key][k + 1] = r[key]
        if r["restore"]:
            idx = k + 1 - np.array(r["restore"])
            y_hist[idx] = y_meas[idx]
            out["flagged"][idx] = False
            for j in range(idx.min() + 1, k + 1):     # a missing sample inside the run was predicted from the
                if np.isnan(y_meas[j]):               # history before the restore: predict it again
                    y_hist[j] = ident.predict(regressor(y_hist, u, j - 1))
            ident.learn(np.array([regressor(y_hist, u, j - 1) for j in idx]), y_meas[idx], r["restore"])
    out["y_clean"] = y_hist
    return out
