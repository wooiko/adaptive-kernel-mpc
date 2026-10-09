"""Building blocks of the demo: commissioning, tuning, threshold calibration, scoring."""
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV

import data
from adaptive import (LAG, AdaptiveIdentifier, Scaler, fit_svr, regression_set, regressor,
                      robust_std, run_online)
from kernel import KRR, loo_grid_search

SIGMAS = np.logspace(-0.3, 1.5, 10)   # kernel width, in normalised units
LAMS = np.logspace(-5, -1, 9)         # regularisation
DEFAULT_SIGMA, DEFAULT_LAM = 3.0, 1e-2
WINDOW = 1000        # sliding-window length, samples
REFIT_EVERY = 20     # accepted samples between refits of the window model
BOOT_THR = 3.5       # first-pass cleaning only: threshold in in-sample robust std of the SVR residual
ALPHA = 0.005        # target share of normal samples rejected by the filter
ALPHAS = (0.01, 0.005, 0.002)   # values compared in the report
HORIZON = 20         # samples (2 min) along which the supervisor checks the planned trajectory


def tune(Z, tz, n_max=1000, seed=0, tol=0.02):
    """Leave-one-out grid search on a random subset of the rows.
    The LOO surface is flat near its minimum, so among all settings within `tol` of
    the best one we take the strongest regularisation and then the narrowest kernel:
    the smoothest model whose uncertainty still grows quickly away from the data."""
    idx = np.random.default_rng(seed).choice(len(Z), size=min(n_max, len(Z)), replace=False)
    _, _, table = loo_grid_search(Z[idx], tz[idx], SIGMAS, LAMS)
    ok = table <= table.min() * (1.0 + tol)
    j = max(np.where(ok.any(0))[0])
    i = min(np.where(ok[:, j])[0])
    return float(SIGMAS[i]), float(LAMS[j]), table, idx


def calibrate_threshold(y, u, Zc, tzc, sc, sigma, noise, alphas=ALPHAS, window=WINDOW, step=REFIT_EVERY):
    """Thresholds of both detectors for target false-rejection shares `alphas`, from
    start-up data only.

    SVR filter, replayed as online: an SVR trained on the `window` cleaned rows before
    row s predicts rows s .. s+step-1, then the window moves on by `step` rows (the online
    filter is refitted every `step` accepted samples). On rows whose target and lagged
    outputs are not spikes (offline centred-median mask) the ratio |residual| / in-sample
    robust std is collected; thr(alpha) is its (1 - alpha) quantile. This corrects both
    the in-sample underestimate of the scatter and its heavy tails.
    Rolling median (reference), causal and centred: (1 - alpha) quantile of the distance
    from the median on the clean samples, in mol/L."""
    spike = data.spike_mask_offline(y, noise) | np.isnan(y)
    X, t, ks = regression_set(y, u)
    clean_row = ~spike[ks] & ~spike[ks - 1] & ~spike[ks - 2]
    ratios = []
    for s0 in range(window, len(Zc), step):
        tr = slice(s0 - window, s0)
        svr = fit_svr(Zc[tr], tzc[tr], sigma, noise / sc.st)
        s_in = robust_std(tzc[tr] - svr.predict(Zc[tr]))
        m = clean_row & (ks >= s0 + LAG) & (ks < min(s0 + step, len(Zc)) + LAG)   # row i <-> target i + LAG
        if m.any():
            ratios.append(np.abs(sc.t(t[m]) - svr.predict(sc.x(X[m]))) / s_in)
    ratios = np.concatenate(ratios)
    med = {}
    for name, causal in (("causal", True), ("centred", False)):
        dev = data.rolling_median_dev(y, causal=causal)[~spike]
        med[name] = {a: float(np.quantile(dev, 1 - a)) for a in alphas}
    return {a: float(np.quantile(ratios, 1 - a)) for a in alphas}, med, int(len(ratios))


def commission(y, u, seed=0):
    """Start-up on a block of historical data.
    1. Bootstrap: scaler, SVR filter and a first KRR from a random subset of raw rows
       (default kernel settings; the SVR loss is robust to the spikes still in there).
    2. Pass the block through the filter once to clean it.
    3. Refit the scaler on the cleaned data; tune (sigma, lam) by leave-one-out.
    4. Calibrate the anomaly thresholds on out-of-sample residuals.
    Returns everything needed to continue online."""
    noise = data.noise_std_estimate(y)
    X, t, _ = regression_set(y, u)
    sc0 = Scaler().fit(X, t)
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=min(WINDOW, len(X)), replace=False))
    boot = AdaptiveIdentifier(DEFAULT_SIGMA, DEFAULT_LAM, sc0, sc0.x(X[idx]), sc0.t(t[idx]), noise,
                              window=WINDOW, thr=BOOT_THR, adapt=False)
    y_init = np.where(np.isnan(y[:LAG]), np.nanmedian(y), y[:LAG])
    first = run_online(boot, y, u, y_init)
    Xc, tc, _ = regression_set(first["y_clean"], u)
    sc = Scaler().fit(Xc, tc)
    Zc, tzc = sc.x(Xc), sc.t(tc)
    sigma, lam, table, idx = tune(Zc, tzc, seed=seed)
    thr, thr_median, n_cal = calibrate_threshold(y, u, Zc, tzc, sc, sigma, noise)
    return {"scaler": sc, "noise": noise, "sigma": sigma, "lam": lam, "loo_table": table,
            "Z": Zc, "tz": tzc, "tune_idx": idx, "y_clean": first["y_clean"], "u": u,
            "flagged": first["flagged"], "Xc": Xc, "tc": tc,
            "thr": thr, "thr_median": thr_median, "n_calibration": n_cal}


def identifier(c, alpha=ALPHA, **kw):
    kw = {"window": WINDOW, "refit_every": REFIT_EVERY, "thr": c["thr"][alpha], **kw}
    return AdaptiveIdentifier(c["sigma"], c["lam"], c["scaler"], c["Z"], c["tz"], c["noise"], **kw)


def first_values(y, fallback):
    """History for the first LAG samples of a record: its own measurements, a missing
    one replaced by `fallback`."""
    y0 = np.array(y[:LAG], float)
    y0[np.isnan(y0)] = fallback
    return y0


class StaticKRR:
    """KRR fitted once on the tuning subset, in physical units."""
    def __init__(self, c):
        self.sc = c["scaler"]
        self.m = KRR(c["sigma"], c["lam"]).fit(c["Z"][c["tune_idx"]], c["tz"][c["tune_idx"]])

    def predict(self, X): return self.sc.t_inv(self.m.predict(self.sc.x(np.atleast_2d(X))))
    def variance(self, X): return self.m.variance(self.sc.x(np.atleast_2d(X)))

    def gradient(self, x):
        """dy/dx in physical units (mol/L per regressor unit), for linearisation in MPC:
        dy/dx_j = (st / sx_j) * df/dz_j."""
        return self.m.gradient(self.sc.x(np.asarray(x, float))) * self.sc.st / self.sc.sx


def fit_arx(c):
    """Linear reference model on the same regressors and the same cleaned data
    (equation error; noise in the lagged outputs biases its coefficients)."""
    return RidgeCV(alphas=np.logspace(-8, 0, 17)).fit(c["Xc"], c["tc"])


class LinearOE:
    """Linear output-error model: same structure as ARX, coefficients fitted to
    minimise the free-run (simulation) error on the cleaned start-up data."""
    def __init__(self, c=None, p=None):
        if p is not None:
            self.p = p
            return
        arx, y, u = fit_arx(c), c["y_clean"], c["u"]
        self.p = least_squares(lambda q: free_run(LinearOE(p=q), y, u) - y,
                               np.r_[arx.coef_, arx.intercept_], x_scale="jac").x

    def predict(self, X): return np.atleast_2d(X) @ self.p[:-1] + self.p[-1]


def free_run(model, y_init, u):
    """Multi-step simulation: the model is fed its OWN past predictions, never the
    measured output. This is what a controller or a what-if study relies on."""
    y_hat = np.empty(len(u))
    y_hat[:LAG] = y_init[:LAG]
    for k in range(LAG - 1, len(u) - 1):
        y_hat[k + 1] = model.predict(regressor(y_hat, u, k)[None, :])[0]
    return y_hat


class Frozen:
    """A stored copy of the window KRR, in physical units."""
    def __init__(self, krr, sc): self.krr, self.sc = krr, sc
    def predict(self, X): return self.sc.t_inv(self.krr.predict(self.sc.x(np.atleast_2d(X))))


def rmse(a, b, mask=None):
    e = (a - b) if mask is None else (a - b)[mask]
    return float(np.sqrt(np.nanmean(e ** 2)))


def fit_percent(y, y_hat):
    return float(100.0 * (1.0 - np.linalg.norm(y - y_hat) / np.linalg.norm(y - y.mean())))


def precision_recall(flag, truth):
    tp = int(np.sum(flag & truth))
    return {"flagged": int(flag.sum()), "true_anomalies": int(truth.sum()), "caught": tp,
            "precision": tp / max(int(flag.sum()), 1), "recall": tp / max(int(truth.sum()), 1),
            "false_rejection_share": int(np.sum(flag & ~truth)) / max(int(np.sum(~truth)), 1)}
