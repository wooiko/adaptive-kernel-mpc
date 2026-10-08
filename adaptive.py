"""Three-level adaptive identification, one sample at a time.

    level 1  SVR filter      rejects anomalous measurements
    level 2  KRR on a sliding window   follows slow drift of the process
    level 3  GP supervisor   measures how far the operating point is from the data
                             the model was trained on, and makes the controller cautious

All three use the same RBF kernel. The model is NARX:
    y[k+1] = f( y[k], y[k-1], u[k], u[k-1] )
"""
from collections import deque

import numpy as np
from sklearn.svm import SVR

from kernel import KRR

NA, NB = 2, 2            # past outputs / past inputs in the regressor
LAG = max(NA, NB)


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
    cautious      th1 <= var < th2    R = R0 * (1 + var / th2)
    conservative  var >= th2          R = 2 R0
    with th1 = lam (the regularisation level) and th2 = 3 th1. Stepping back to a
    calmer regime requires `hold` consecutive samples below the threshold."""
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
        mult = (1.0, 1.0 + min(var / self.th2, 1.0), 2.0)[self.regime]
        return self.regime, mult


class AdaptiveIdentifier:
    def __init__(self, sigma, lam, scaler, Z0, tz0, noise_std, window=500, kappa=2.5,
                 n_jump=4, use_filter=True, adapt=True, refit_every=10):
        self.sigma, self.lam, self.sc = sigma, lam, scaler
        self.kappa, self.n_jump = kappa, n_jump
        self.use_filter, self.adapt, self.refit_every = use_filter, adapt, refit_every
        self.noise_z = noise_std / scaler.st
        self.Z, self.tz = deque(Z0[-window:], maxlen=window), deque(tz0[-window:], maxlen=window)
        self.krr = KRR(sigma, lam).fit(np.array(self.Z), np.array(self.tz))
        self.supervisor = Supervisor(lam)
        self._consec, self._jump, self._passed, self._since = 0, False, 0, 0
        self._refit_svr()

    def _refit_svr(self):
        Z, tz = np.array(self.Z), np.array(self.tz)
        self.svr = fit_svr(Z, tz, self.sigma, self.noise_z)
        self.res_std = robust_std(tz - self.svr.predict(Z))   # normal one-step scatter

    def step(self, x, y_meas):
        """x: regressor built from data up to k; y_meas: measurement of y[k+1] (may be NaN).
        Returns prediction, anomaly flag, the value to keep in the history, GP variance,
        regime and penalty multiplier."""
        z = self.sc.x(x)[None, :]
        pred = float(self.sc.t_inv(self.krr.predict(z)[0]))
        var = float(self.krr.variance(z)[0])
        regime, mult = self.supervisor.update(var)
        missing = np.isnan(y_meas)
        flagged = False
        if not missing and self.use_filter:
            delta = abs(self.sc.t(y_meas) - self.svr.predict(z)[0])
            over = delta > self.kappa * self.res_std
            if self._jump:
                # A run of anomalies means the process itself has changed: accept the
                # new data until the refitted filter agrees with them again.
                self._passed = 0 if over else self._passed + 1
                if self._passed >= self.n_jump:
                    self._jump, self._consec = False, 0
            elif over:
                self._consec += 1
                if self._consec >= self.n_jump:
                    self._jump, self._passed = True, 0
                else:
                    flagged = True
            else:
                self._consec = 0
        accept = not missing and not flagged
        if accept and self.adapt:
            self.Z.append(z[0]); self.tz.append(self.sc.t(y_meas))
            self._since += 1
            if self._jump or self._since >= self.refit_every:
                self.krr.fit(np.array(self.Z), np.array(self.tz))
                if self.use_filter:
                    self._refit_svr()
                self._since = 0
        return {"pred": pred, "flagged": flagged, "jump": self._jump,
                "y_keep": y_meas if accept else pred, "var": var, "regime": regime, "mult": mult}


def run_online(ident, y_meas, u, y_init):
    """Feed a record through the identifier. The regressor always uses the cleaned
    history (rejected or missing samples are replaced by the model prediction)."""
    n = len(u)
    y_hist = np.array(y_meas, dtype=float)
    y_hist[:LAG] = y_init[:LAG]
    out = {k: np.full(n, np.nan) for k in ("pred", "var", "mult")}
    out["flagged"], out["regime"] = np.zeros(n, bool), np.zeros(n, int)
    for k in range(LAG - 1, n - 1):
        r = ident.step(regressor(y_hist, u, k), y_meas[k + 1])
        y_hist[k + 1] = r["y_keep"]
        for key in ("pred", "var", "mult", "flagged", "regime"):
            out[key][k + 1] = r[key]
    out["y_clean"] = y_hist
    return out
